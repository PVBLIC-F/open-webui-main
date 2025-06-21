"""
Bot Launcher for Open WebUI

Simple launcher that starts the bot as a subprocess to avoid import conflicts
and ensure the main application doesn't break if the bot has issues.
"""

import subprocess
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def start_bot_worker():
    """
    Start the bot worker as a subprocess.
    
    This approach is safer than importing the bot directly because:
    - Avoids import conflicts
    - Bot errors don't crash the main app  
    - Can restart independently
    - Works with Render deployment
    
    Returns:
        subprocess.Popen: Bot process handle, or None if startup failed
    """
    try:
        # Get path to bot framework
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        bot_framework_path = project_root / "bot-framework"
        bot_script = bot_framework_path / "multi_provider_bot.py"
        
        if not bot_script.exists():
            logger.warning(f"Bot script not found at {bot_script}")
            return None
        
        # Check if bot is already running (cross-platform)
        if _is_bot_running():
            logger.info("Bot is already running")
            return None
        
        # Load environment variables
        env = _load_bot_environment(bot_framework_path)
        
        # Start bot as subprocess
        logger.info(f"Starting bot worker from {bot_script}")
        
        process = subprocess.Popen(
            [sys.executable, str(bot_script)],
            cwd=str(bot_framework_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        logger.info(f"Bot worker started with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Error starting bot worker: {e}")
        return None

def _is_bot_running() -> bool:
    """Check if bot is already running (cross-platform)"""
    try:
        # Try pgrep on Unix systems
        if os.name != 'nt':
            result = subprocess.run(
                ["pgrep", "-f", "multi_provider_bot.py"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        else:
            # Use tasklist on Windows
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True
            )
            return "multi_provider_bot.py" in result.stdout
    except:
        # If process checking fails, assume not running
        return False

def _load_bot_environment(bot_framework_path: Path) -> dict:
    """Load environment variables for bot subprocess"""
    env = os.environ.copy()
    env_file = bot_framework_path / ".env"
    
    if env_file.exists():
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(env_file)
            env.update(env_vars)
            logger.info(f"Loaded {len(env_vars)} environment variables")
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env file")
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
    
    return env 