"""
Automatic Model Logo Detection using Lobe Icons
https://github.com/lobehub/lobe-icons

This module provides automatic logo detection for AI/LLM models by mapping
model IDs and names to their corresponding Lobe Icons slugs.
"""

import re
from typing import Optional

# Model pattern to Lobe Icons slug mapping
# Based on the official Lobe Icons documentation: https://lobehub.com/icons
# 
# ONLY includes icons that actually exist in Lobe Icons
# Patterns are matched against model IDs and names (case-insensitive)
# Longer/more specific patterns are matched first for better precision
MODEL_LOGO_MAP = {
    # === MODELS (from Lobe Icons documentation) ===
    
    # OpenAI
    'openai': 'openai',
    'chatgpt': 'openai',
    'gpt': 'openai',
    'dall-e': 'dalle',
    'dalle': 'dalle',
    'whisper': 'openai',
    'o1': 'openai',
    
    # Anthropic
    'anthropic': 'anthropic',
    'claude': 'claude',
    
    # Google
    'gemini': 'gemini',
    'gemma': 'gemma',
    'palm': 'palm',
    'google': 'google',
    'vertex': 'vertexai',
    'vertexai': 'vertexai',
    'aistudio': 'aistudio',
    'deepmind': 'deepmind',
    'colab': 'colab',
    'notebooklm': 'notebooklm',
    
    # Meta
    'meta': 'meta',
    'llama': 'meta',
    'metaai': 'metaai',
    'metagpt': 'metagpt',
    
    # Microsoft
    'githubcopilot': 'githubcopilot',
    'copilot': 'copilot',
    'azure': 'azure',
    'azureai': 'azureai',
    'bing': 'bing',
    'microsoft': 'microsoft',
    
    # Mistral
    'mistral': 'mistral',
    'mixtral': 'mistral',
    
    # Other Major Models
    'ai21': 'ai21',
    'jamba': 'ai21',
    'aionlabs': 'aionlabs',
    'assemblyai': 'assemblyai',
    'aya': 'aya',
    'baai': 'baai',
    'aquila': 'baai',
    'baichuan': 'baichuan',
    'chatglm': 'chatglm',
    'codegeex': 'codegeex',
    'cogvideo': 'cogvideo',
    'cogview': 'cogview',
    'commanda': 'commanda',
    'dbrx': 'dbrx',
    'deepseek': 'deepseek',
    'doubao': 'doubao',
    'fishaudio': 'fishaudio',
    'flux': 'flux',
    'grok': 'grok',
    'hunyuan': 'hunyuan',
    'inflection': 'inflection',
    'internlm': 'internlm',
    'liquid': 'liquid',
    'llava': 'llava',
    'magic': 'magic',
    'minimax': 'minimax',
    'nova': 'nova',
    'openchat': 'openchat',
    'qwen': 'qwen',
    'rwkv': 'rwkv',
    'sensenova': 'sensenova',
    'spark': 'spark',
    'stepfun': 'stepfun',
    'voyage': 'voyage',
    'wenxin': 'wenxin',
    'xuanyuan': 'xuanyuan',
    'yi': 'yi',
    
    # === PROVIDERS (from Lobe Icons documentation) ===
    
    'zeroone': 'zeroone',
    '01-ai': 'zeroone',
    '01.ai': 'zeroone',
    'ai360': 'ai360',
    'aihubmix': 'aihubmix',
    'aimass': 'aimass',
    'alephalpha': 'alephalpha',
    'alibaba': 'alibaba',
    'alibabacloud': 'alibabacloud',
    'antgroup': 'antgroup',
    'anyscale': 'anyscale',
    'aws': 'aws',
    'baidu': 'baidu',
    'baiducloud': 'baiducloud',
    'baseten': 'baseten',
    'bedrock': 'bedrock',
    'burncloud': 'burncloud',
    'bytedance': 'bytedance',
    'centml': 'centml',
    'cerebras': 'cerebras',
    'civitai': 'civitai',
    'cloudflare': 'cloudflare',
    'cohere': 'cohere',
    'crusoe': 'crusoe',
    'deepinfra': 'deepinfra',
    'exa': 'exa',
    'fal': 'fal',
    'featherless': 'featherless',
    'fireworks': 'fireworks',
    'friendli': 'friendli',
    'giteeai': 'giteeai',
    'github': 'github',
    'groq': 'groq',
    'higress': 'higress',
    'huggingface': 'huggingface',
    'hf': 'huggingface',
    'hyperbolic': 'hyperbolic',
    'iflytekcloud': 'iflytekcloud',
    'inference': 'inference',
    'infermatic': 'infermatic',
    'infinigence': 'infinigence',
    'jina': 'jina',
    'kluster': 'kluster',
    'lambda': 'lambda',
    'leptonai': 'leptonai',
    'lmstudio': 'lmstudio',
    'lobehub': 'lobehub',
    'modelscope': 'modelscope',
    'moonshot': 'moonshot',
    'kimi': 'kimi',
    'nplcloud': 'nplcloud',
    'nebius': 'nebius',
    'nousresearch': 'nousresearch',
    'novita': 'novita',
    'nvidia': 'nvidia',
    'ollama': 'ollama',
    'openrouter': 'openrouter',
    'parasail': 'parasail',
    'perplexity': 'perplexity',
    'pplx': 'perplexity',
    'ppio': 'ppio',
    'qiniu': 'qiniu',
    'replicate': 'replicate',
    'sambanova': 'sambanova',
    'search1api': 'search1api',
    'searchapi': 'searchapi',
    'siliconcloud': 'siliconcloud',
    'snowflake': 'snowflake',
    'stability': 'stability',
    'stable-diffusion': 'stability',
    'statecloud': 'statecloud',
    'targon': 'targon',
    'tencent': 'tencent',
    'tencentcloud': 'tencentcloud',
    'tii': 'tii',
    'together': 'together',
    'upstage': 'upstage',
    'vercel': 'vercel',
    'vllm': 'vllm',
    'volcengine': 'volcengine',
    'workersai': 'workersai',
    'xai': 'xai',
    'xinference': 'xinference',
    'yandex': 'yandex',
    'zhipu': 'zhipu',
    
    # === APPLICATIONS (from Lobe Icons documentation) ===
    
    'adobe': 'adobe',
    'adobefirefly': 'adobefirefly',
    'firefly': 'adobefirefly',
    'automatic': 'automatic',
    'cline': 'cline',
    'clipdrop': 'clipdrop',
    'comfyui': 'comfyui',
    'coqui': 'coqui',
    'coze': 'coze',
    'crewai': 'crewai',
    'cursor': 'cursor',
    'deepai': 'deepai',
    'dify': 'dify',
    'doc2x': 'doc2x',
    'docsearch': 'docsearch',
    'dreammachine': 'dreammachine',
    'fastgpt': 'fastgpt',
    'flora': 'flora',
    'flowith': 'flowith',
    'glif': 'glif',
    'goose': 'goose',
    'gradio': 'gradio',
    'greptile': 'greptile',
    'hailuo': 'hailuo',
    'haiper': 'haiper',
    'hedra': 'hedra',
    'ideogram': 'ideogram',
    'jimeng': 'jimeng',
    'kera': 'kera',
    'kling': 'kling',
    'langchain': 'langchain',
    'langfuse': 'langfuse',
    'langgraph': 'langgraph',
    'langsmith': 'langsmith',
    'lightricks': 'lightricks',
    'livekit': 'livekit',
    'llamaindex': 'llamaindex',
    'luma': 'luma',
    'make': 'make',
    'manus': 'manus',
    'mcp': 'mcp',
    'midjourney': 'midjourney',
    'mj': 'midjourney',
    'monica': 'monica',
    'myshell': 'myshell',
    'n8n': 'n8n',
    'notion': 'notion',
    'novelai': 'novelai',
    'openwebui': 'openwebui',
    'phidata': 'phidata',
    'pika': 'pika',
    'pixverse': 'pixverse',
    'player2': 'player2',
    'poe': 'poe',
    'pollinations': 'pollinations',
    'pydanticai': 'pydanticai',
    'qingyan': 'qingyan',
    'railway': 'railway',
    'recraft': 'recraft',
    'replit': 'replit',
    'rsshub': 'rsshub',
    'runway': 'runway',
    'suno': 'suno',
    'sync': 'sync',
    'tiangong': 'tiangong',
    'topazlabs': 'topazlabs',
    'trae': 'trae',
    'tripo': 'tripo',
    'udio': 'udio',
    'unstructured': 'unstructured',
    'v0': 'v0',
    'vectorizerai': 'vectorizerai',
    'vidu': 'vidu',
    'viggle': 'viggle',
    'yuanbao': 'yuanbao',
    'zapier': 'zapier',
    'zeabur': 'zeabur'
}


def detect_model_logo(model_id: str, model_name: Optional[str] = None) -> Optional[str]:
    """
    Detect the appropriate logo for a model based on its ID or name
    
    Args:
        model_id: The model ID (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet")
        model_name: Optional model name for additional matching
    
    Returns:
        The Lobe Icons slug or None if no match found
    """
    if not model_id:
        return None
    
    normalized_id = model_id.lower()
    normalized_name = model_name.lower() if model_name else ''
    
    # Try pattern matching first - sort by length (longer patterns first) for better precision
    sorted_patterns = sorted(MODEL_LOGO_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    
    for pattern, logo_slug in sorted_patterns:
        if pattern in normalized_id or pattern in normalized_name:
            return logo_slug
    
    # Fallback to provider prefix matching (e.g., "openai/gpt-4" -> "openai")
    provider_match = re.match(r'^([^/]+)/', normalized_id)
    if provider_match:
        provider = provider_match.group(1)
        if provider in MODEL_LOGO_MAP:
            return MODEL_LOGO_MAP[provider]
    
    return None


def get_model_profile_image_url(
    model_id: str,
    model_name: str = None,
    custom_url: str = None,
    fallback_url: str = "/static/favicon.png"
) -> str:
    """
    Get the best profile image URL for a model
    Prioritizes custom logos over auto-detected ones, but prefers auto-detection over default favicon
    
    Args:
        model_id: The model ID
        model_name: Optional model name
        custom_url: Custom profile image URL from model metadata
        fallback_url: Default fallback URL
        
    Returns:
        The best available profile image URL
    """
    # If we have a custom URL that's not the default favicon, use it
    if custom_url and custom_url != "/static/favicon.png" and "favicon.png" not in custom_url:
        return custom_url
    
    # Try to auto-detect a logo
    logo_slug = detect_model_logo(model_id, model_name)
    if logo_slug:
        return f"https://unpkg.com/@lobehub/icons-static-svg@latest/icons/{logo_slug}.svg"
    
    # Fallback to the provided fallback URL
    return fallback_url 