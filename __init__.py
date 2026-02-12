import asyncio
import json
import io
import base64
import hashlib
import random
import numpy as np
import httpx  # 改用 httpx 库，需要安装: pip install httpx
from PIL import Image
from pathlib import Path
import folder_paths

class LLMChatNode:
    """
    ComfyUI Custom Node for calling OpenAI-compatible APIs.
    Switched to 'httpx' library (thread-based) to fix 30s+ timeout issues with Thinking models.
    """
    
    # 保持缓存目录在当前节点文件夹内
    MODEL_CACHE_DIR = Path(__file__).parent / "llm_chat_cache"
    LAST_URL_FILE = MODEL_CACHE_DIR / "last_base_url.txt"
    
    def __init__(self):
        self._result_cache = {}
    
    @classmethod
    def _cache_dir_for_base(cls, base_url: str | None):
        """
        根据 base_url 生成独立缓存目录，使用 MD5 避免路径非法字符。
        """
        base_key = (base_url or "default").strip().lower()
        base_hash = hashlib.md5(base_key.encode()).hexdigest()
        return cls.MODEL_CACHE_DIR / base_hash
    
    @classmethod
    def _cache_file_for_base(cls, base_url: str | None):
        return cls._cache_dir_for_base(base_url) / "models.json"
    
    @classmethod
    def _load_model_cache_static(cls, base_url: str | None = None):
        cache_file = cls._cache_file_for_base(base_url)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load model cache: {e}")
        return []

    @classmethod
    def _get_last_base_url(cls):
        if cls.LAST_URL_FILE.exists():
            try:
                return cls.LAST_URL_FILE.read_text(encoding="utf-8").strip()
            except:
                pass
        return None

    @classmethod
    def _set_last_base_url(cls, base_url: str):
        try:
            cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cls.LAST_URL_FILE.write_text(base_url, encoding="utf-8")
        except Exception as e:
            print(f"[WARNING] Failed to save last base_url: {e}")

    @classmethod
    def _load_filtered_models(cls):
        """
        根据最后使用的 base_url 加载模型列表。
        如果没记录，则回退到加载所有已缓存的模型。
        """
        last_url = cls._get_last_base_url()
        if last_url:
            models = cls._load_model_cache_static(last_url)
            if models:
                print(f"[INFO] Loading models for last used base_url: {last_url}")
                return models

        # 如果没有针对特定 URL 的缓存，则加载全部
        seen = set()
        ordered_models = []
        def add_models(models):
            for m in models:
                if m not in seen:
                    seen.add(m)
                    ordered_models.append(m)

        if cls.MODEL_CACHE_DIR.exists():
            for p in cls.MODEL_CACHE_DIR.iterdir():
                if not p.is_dir():
                    continue
                cache_file = p / "models.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            add_models(json.load(f))
                    except:
                        pass
        return ordered_models
    
    def _save_model_cache(self, models, base_url: str | None):
        cache_dir = self._cache_dir_for_base(base_url)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "models.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(models, f)
        except Exception as e:
            print(f"[WARNING] Failed to save model cache: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_list = cls._load_filtered_models()
        
        last_url = cls._get_last_base_url() or "https://api.openai.com/v1"

        if not model_list:
            model_list = ["perplexity-search"]
            print("[INFO] No cache found, using default model list.")
        else:
            print(f"[INFO] Using {len(model_list)} models from cache")
        
        return {
            "required": {
                "base_url": ("STRING", {"default": last_url}),
                "api_key": ("STRING", {"default": ""}),
                "model": (model_list, {"default": model_list[0]}),
                "refresh_models": ("BOOLEAN", {"default": False}),
                "incognito": ("BOOLEAN", {"default": False, "label_on": "Incognito ON", "label_off": "Incognito OFF"}),
                "timeout": ("INT", {"default": 600, "min": 10, "max": 36000, "step": 10, "display": "number"}),
                "always_rerun": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Always Execute (Fresh Results)",
                    "label_off": "Use Cache (Save Tokens)"
                }),
                "use_random_seed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "fail_words": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {
                    "default": "You are an image description generation expert...",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {"default": "Describe the image.", "multiline": True}),
            },
            "optional": {"image": ("IMAGE",)}
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("thinking", "response", "raw_json")
    FUNCTION = "execute"
    CATEGORY = "Utils/AI"
    
    def _image_hash(self, image):
        if image is None:
            return "no_image"
        i = 255.0 * image.cpu().numpy().squeeze()
        return hashlib.md5(i.tobytes()).hexdigest()
    
    def _generate_cache_key_ignore_seed(self, base_url, model, system_prompt, user_prompt, fail_words, image, incognito):
        image_hash = self._image_hash(image)
        if isinstance(fail_words, dict):
            fail_words_str = str(fail_words.get("value", fail_words))
        elif fail_words is None:
            fail_words_str = ""
        else:
            fail_words_str = str(fail_words)
        cache_data = f"{base_url}|{model}|{system_prompt}|{user_prompt}|{fail_words_str}|{image_hash}|{incognito}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _fetch_models_sync(self, base_url, api_key):
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
            else:
                print(f"[ERROR] Failed to fetch models: {resp.status_code} {resp.text}")
                return []
        except Exception as e:
            print(f"[ERROR] Error fetching models: {e}")
            return []
    
    def _call_openai_sync(self, url, headers, data, max_retries, fail_words, timeout):
        fail_words_list = []
        if fail_words:
            if isinstance(fail_words, str):
                fail_words_list = [w.strip().lower() for w in fail_words.split(",") if w.strip()]
            else:
                try:
                    fail_words_str = str(fail_words)
                    fail_words_list = [w.strip().lower() for w in fail_words_str.split(",") if w.strip()]
                except:
                    pass

        # 强制启用 stream
        data["stream"] = True
        
        for attempt in range(max_retries):
            client = httpx.Client()
            try:
                # 记录响应
                full_content = ""
                full_thinking = ""
                last_chunk_raw = None
                
                with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=data,
                    timeout=httpx.Timeout(connect=20.0, read=timeout, write=60.0, pool=20.0)
                ) as response:
                    if response.status_code != 200:
                        error_text = response.read().decode(errors="ignore")
                        print(f"[WARNING] API Error {response.status_code}: {error_text}")
                        if response.status_code >= 500:
                            import time
                            time.sleep(1)
                            continue
                        return f"Error {response.status_code}: {error_text}"
                    
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        
                        raw_data = line[5:].strip()
                        if raw_data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(raw_data)
                            last_chunk_raw = chunk
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                
                                # 处理内容
                                if "content" in delta and delta["content"]:
                                    full_content += delta["content"]
                                
                                # 处理思考过程 (OpenAI standard reasoning_content or some providers' thinking)
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    full_thinking += delta["reasoning_content"]
                                elif "thinking" in delta and delta["thinking"]:
                                    full_thinking += delta["thinking"]
                                    
                        except Exception as e:
                            print(f"[DEBUG] Error parsing SSE line: {e} | Line: {line}")
                            continue

                if fail_words_list and any(fw in full_content.lower() for fw in fail_words_list):
                    print(f"[WARNING] Response contained fail word, retrying ({attempt + 1}/{max_retries})...")
                    continue
                
                # 构造兼容格式的返回结果
                final_result = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        }
                    }]
                }
                if full_thinking:
                    # 如果有独立思考字段，包装进 content 以便后续解析或直接记录
                    if "<think>" not in full_content:
                        final_result["choices"][0]["message"]["content"] = f"<think>\n{full_thinking}\n</think>\n{full_content}"
                
                return final_result
                
            except httpx.TimeoutException:
                print(f"[WARNING] Timeout error attempt {attempt + 1} (Limit: {timeout}s)")
            except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
                print(f"[WARNING] Connection error attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"[WARNING] Unexpected error attempt {attempt + 1}: {e}")
            finally:
                client.close()
            
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
        
        return "Max retries exceeded or connection failed"
    
    async def _call_openai_api_wrapper(self, url, headers, data, max_retries, fail_words, timeout):
        return await asyncio.to_thread(
            self._call_openai_sync, url, headers, data, max_retries, fail_words, timeout
        )
    
    def _parse_response(self, raw_response):
        if isinstance(raw_response, str):
            return "", raw_response
        
        try:
            choice = raw_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            thinking = ""
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>") + 7
                end = content.find("</think>")
                thinking = content[start:end].strip()
                content = content.replace(content[content.find("<think>"):end+8], "").strip()
            
            return thinking, content
        except Exception as e:
            return "", f"Error parsing JSON: {e}"
    
    async def execute(
        self,
        base_url,
        api_key,
        model,
        refresh_models,
        incognito,
        timeout,
        always_rerun,
        use_random_seed,
        seed,
        max_retries,
        fail_words,
        system_prompt,
        user_prompt,
        image=None,
    ):
        # 如果勾选了刷新模型，直接执行获取逻辑，不进行任何前置判断或执行后续逻辑
        if refresh_models:
            print(f"[INFO] Refreshing model list for {base_url}...")
            remote_models = await asyncio.to_thread(self._fetch_models_sync, base_url, api_key)
            if remote_models:
                self._save_model_cache(remote_models, base_url)
                self._set_last_base_url(base_url)
                print(f"[INFO] Saved {len(remote_models)} models to cache and updated last used URL.")
                return ("", "Model list refreshed. Please reload the page or node to update the dropdown list.", "")
            else:
                return ("", "Failed to fetch models. Please check your base_url and API Key.", "")
        
        cache_key = self._generate_cache_key_ignore_seed(
            base_url, model, system_prompt, user_prompt, fail_words, image, incognito
        )
        
        if (not always_rerun) and (cache_key in self._result_cache):
            print(f"[INFO] 💾 Cache hit: {cache_key[:16]}...")
            return self._result_cache[cache_key]
        
        if always_rerun:
            actual_seed = random.randint(0, 0xffffffffffffffff) if use_random_seed else seed
            seed_mode = "random" if use_random_seed else "fixed"
        else:
            actual_seed = seed
            seed_mode = "fixed"
        
        print(f"[INFO] 🔄 FRESH API call | Model: {model} | Seed: {actual_seed} ({seed_mode}) | Incognito: {incognito}")
        
        img_base64 = None
        if image is not None:
            try:
                i = 255.0 * image.cpu().numpy().squeeze()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                raise Exception(f"Error processing image: {e}")
        
        messages = [{"role": "system", "content": system_prompt}]
        user_content = [{"type": "text", "text": user_prompt}]
        if img_base64:
            user_content.append({"type": "image_url", "image_url": {"url": f"image/png;base64,{img_base64}"}})
        messages.append({"role": "user", "content": user_content})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        # 优化消息格式：不带图片时使用纯字符串 content 以提高兼容性
        if not img_base64:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": actual_seed,
            "incognito": incognito
        }
        
        raw_response = await self._call_openai_api_wrapper(url, headers, data, max_retries, fail_words, timeout)
        
        if isinstance(raw_response, str) and ("Error" in raw_response or "Max retries" in raw_response):
            print(f"[ERROR] API call failed: {raw_response}")
            raise Exception(raw_response)
        
        thinking, result = self._parse_response(raw_response)
        print("[INFO] ✅ Request completed")
        
        if not isinstance(result, str):
            result = str(result)
        
        result_tuple = (thinking, result, json.dumps(raw_response) if isinstance(raw_response, dict) else str(raw_response))
        self._result_cache[cache_key] = result_tuple
        
        if len(self._result_cache) > 100:
            del self._result_cache[next(iter(self._result_cache))]
        
        return result_tuple

NODE_CLASS_MAPPINGS = {"LLMChat": LLMChatNode}
NODE_DISPLAY_NAME_MAPPINGS = {"LLMChat": "LLM Chat"}
