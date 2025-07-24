#!/usr/bin/env python3
"""
Lightweight Telegram Bot Manager
A simple, stateless Telegram bot for notifications and occasional command checking.

Features:
- Send notifications instantly
- Check for new messages on-demand (no continuous polling)
- Process commands when you call check_messages()
- Stateless design - no continuous processes needed
- Rate limiting and error handling
- Authorization system
- Persistent state via JSON file
- Table formatting with colors and proper alignment

Usage:
- Call send_notification() anytime to send messages
- Call check_messages() every few seconds/minutes from your main script
- Use send_table() for formatted tabular data
- No background processes or threads required

Author: Sambath S
Date: 2025
"""

import json
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
import requests
from functools import wraps


class TelegramBotManager:
    """
    Lightweight Telegram Bot Manager - No continuous processes required!
    
    This class provides a simple interface for:
    - Sending notifications instantly
    - Checking for new messages on-demand
    - Processing commands when you want
    - Maintaining state between calls
    - Formatting and sending tables with colors
    """
    
    def __init__(self, bot_token: str, authorized_chat_ids: List[int], 
                 state_file: str = "bot_state.json", log_file: Optional[str] = None):
        """
        Initialize the Telegram Bot Manager.
        
        Args:
            bot_token: Your Telegram bot token from @BotFather
            authorized_chat_ids: List of chat IDs allowed to interact with the bot
            state_file: JSON file to store bot state (last_update_id, etc.)
            log_file: Optional file path for logging
        """
        self.bot_token = bot_token
        self.authorized_chat_ids = set(authorized_chat_ids)
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.state_file = state_file
        
        # Rate limiting - simple timestamp tracking
        self.last_api_call = 0
        self.min_call_interval = 0.04  # ~25 calls per second max
        
        # Command handlers
        self.command_handlers: Dict[str, Callable] = {}
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Load/initialize state
        self.load_state()
        
        # Register default commands
        self.register_default_commands()
        
        self.logger.info("Lightweight TelegramBotManager initialized")
    
    def setup_logging(self, log_file: Optional[str]):
        """Setup logging configuration"""
        self.logger = logging.getLogger('TelegramBot')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup file logging: {e}")
    
    def load_state(self):
        """Load bot state from JSON file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_update_id = state.get('last_update_id', 0)
                    self.logger.info(f"Loaded state: last_update_id={self.last_update_id}")
            else:
                self.last_update_id = 0
                self.logger.info("No state file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            self.last_update_id = 0
    
    def save_state(self):
        """Save bot state to JSON file"""
        try:
            state = {
                'last_update_id': self.last_update_id,
                'last_save': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _rate_limit(self):
        """Simple rate limiting - wait if needed"""
        now = time.time()
        time_since_last = now - self.last_api_call
        if time_since_last < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def is_authorized(self, chat_id: int) -> bool:
        """Check if a chat ID is authorized"""
        return chat_id in self.authorized_chat_ids
    
    def send_notification(self, message: str, chat_id: Optional[int] = None, 
                         parse_mode: str = 'HTML', silent: bool = False) -> bool:
        """
        Send a notification message (main method you'll use from your trading script).
        
        Args:
            message: The message to send
            chat_id: Specific chat ID (if None, sends to all authorized chats)
            parse_mode: Message formatting ('HTML', 'Markdown', or None)
            silent: Send silently (no notification sound)
            
        Returns:
            bool: True if sent successfully to at least one chat
        """
        if chat_id:
            return self._send_message_to_chat(chat_id, message, parse_mode, silent)
        else:
            # Send to all authorized chats
            success_count = 0
            for chat_id in self.authorized_chat_ids:
                if self._send_message_to_chat(chat_id, message, parse_mode, silent):
                    success_count += 1
            return success_count > 0

    def send_image(self, image_path: str, caption: str = "", chat_id: Optional[int] = None, 
                   silent: bool = False) -> bool:
        """
        Send an image file to Telegram.
        
        Args:
            image_path: Path to the image file to send
            caption: Optional caption for the image
            chat_id: Specific chat ID (if None, sends to all authorized chats)
            silent: Send silently (no notification sound)
            
        Returns:
            bool: True if sent successfully to at least one chat
        """
        if chat_id:
            return self._send_image_to_chat(chat_id, image_path, caption, silent)
        else:
            # Send to all authorized chats
            success_count = 0
            for chat_id in self.authorized_chat_ids:
                if self._send_image_to_chat(chat_id, image_path, caption, silent):
                    success_count += 1
            return success_count > 0
    
    def _send_message_to_chat(self, chat_id: int, text: str, parse_mode: str, silent: bool) -> bool:
        """Internal method to send message to specific chat"""
        if not self.is_authorized(chat_id):
            self.logger.warning(f"Attempted to send to unauthorized chat: {chat_id}")
            return False
        
        self._rate_limit()
        
        # Handle long messages
        if len(text) > 4096:
            return self._send_long_message(chat_id, text, parse_mode, silent)
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': text,
            'disable_notification': silent
        }
        
        if parse_mode:
            payload['parse_mode'] = parse_mode
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Notification sent to {chat_id}: {text[:50]}...")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send notification to {chat_id}: {e}")
            return False
    
    def _send_long_message(self, chat_id: int, text: str, parse_mode: str, silent: bool) -> bool:
        """Handle messages longer than 4096 characters"""
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
        success = True
        
        for chunk in chunks:
            if not self._send_message_to_chat(chat_id, chunk, parse_mode, silent):
                success = False
            time.sleep(0.1)  # Small delay between chunks
        
        return success
    
    def _send_image_to_chat(self, chat_id: int, image_path: str, caption: str, silent: bool) -> bool:
        """Internal method to send image to specific chat"""
        if not self.is_authorized(chat_id):
            self.logger.warning(f"Attempted to send image to unauthorized chat: {chat_id}")
            return False
        
        import os
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return False
        
        self._rate_limit()
        
        url = f"{self.base_url}/sendPhoto"
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'photo': image_file}
                data = {
                    'chat_id': chat_id,
                    'disable_notification': silent
                }
                
                if caption:
                    data['caption'] = caption
                    data['parse_mode'] = 'HTML'
                
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
                
                self.logger.info(f"Image sent to {chat_id}: {image_path}")
                return True
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send image to {chat_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending image to {chat_id}: {e}")
            return False
    
    def check_messages(self, timeout: int = 5) -> int:
        """
        Check for new messages and process them (call this from your main loop).
        
        Args:
            timeout: How long to wait for new messages (keep low for quick checks)
            
        Returns:
            int: Number of messages processed
        """
        self._rate_limit()
        
        url = f"{self.base_url}/getUpdates"
        payload = {
            'offset': self.last_update_id + 1,
            'timeout': timeout,
            'limit': 10  # Process up to 10 messages at once
        }
        
        try:
            response = requests.get(url, params=payload, timeout=timeout + 2)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('ok'):
                self.logger.error(f"Telegram API error: {data}")
                return 0
            
            updates = data.get('result', [])
            processed = 0
            
            for update in updates:
                self._process_update(update)
                processed += 1
                self.last_update_id = update['update_id']
            
            if processed > 0:
                self.save_state()  # Save state after processing messages
                self.logger.info(f"Processed {processed} messages")
            
            return processed
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to check messages: {e}")
            return 0
    
    def _process_update(self, update: Dict):
        """Process a single update"""
        try:
            if 'message' in update:
                self._process_message(update['message'])
        except Exception as e:
            self.logger.error(f"Error processing update: {e}")
    
    def _process_message(self, message: Dict):
        """Process an incoming message"""
        chat_id = message['chat']['id']
        user_info = f"{message['chat'].get('first_name', '')} {message['chat'].get('last_name', '')}".strip()
        message_text = message.get('text', '')
        
        # Log all messages
        if self.is_authorized(chat_id):
            self.logger.info(f"Message from {user_info} ({chat_id}): {message_text}")
            
            # Process commands
            if message_text.startswith('/'):
                self._process_command(message)
        else:
            self.logger.warning(f"Unauthorized message from {user_info} ({chat_id}): {message_text}")
    
    def _process_command(self, message: Dict):
        """Process a command message"""
        text = message.get('text', '')
        chat_id = message['chat']['id']
        
        # Parse command and arguments
        parts = text.split()
        command = parts[0].lower().lstrip('/')
        args = parts[1:] if len(parts) > 1 else []
        
        # Remove bot username if present
        if '@' in command:
            command = command.split('@')[0]
        
        self.logger.info(f"Processing command: /{command}")
        
        # Execute command handler
        if command in self.command_handlers:
            try:
                self.command_handlers[command](message, args)
            except Exception as e:
                self.logger.error(f"Error executing command {command}: {e}")
                self.send_notification(f"❌ Error: {str(e)}", chat_id)
        else:
            self.send_notification(f"❓ Unknown command: /{command}", chat_id)
    
    def register_command(self, command: str, handler: Callable):
        """Register a command handler"""
        command = command.lower().lstrip('/')
        self.command_handlers[command] = handler
        self.logger.info(f"Registered command: /{command}")
    
    def register_default_commands(self):
        """Register built-in commands"""
        self.register_command('status', self._cmd_status)
        self.register_command('help', self._cmd_help)
        self.register_command('ping', self._cmd_ping)
        self.register_command('info', self._cmd_info)
    
    # Built-in command handlers
    def _cmd_status(self, message: Dict, args: List[str]):
        """Handle /status command"""
        chat_id = message['chat']['id']
        status_text = f"""
📊 <b>Bot Status</b>

🟢 Status: Active
📱 Mode: On-demand checking
👥 Authorized Users: {len(self.authorized_chat_ids)}
🔧 Commands: {len(self.command_handlers)}
📨 Last Update ID: {self.last_update_id}

<i>Bot ready for notifications!</i>
        """.strip()
        self.send_notification(status_text, chat_id)
    
    def _cmd_help(self, message: Dict, args: List[str]):
        """Handle /help command"""
        chat_id = message['chat']['id']
        commands_list = '\n'.join([f"/{cmd}" for cmd in sorted(self.command_handlers.keys())])
        
        help_text = f"""
🆘 <b>Available Commands:</b>

{commands_list}

💡 <b>How it works:</b>
This bot checks for messages only when your trading script calls check_messages().
Send notifications anytime with send_notification().
        """
        self.send_notification(help_text, chat_id)
    
    def _cmd_ping(self, message: Dict, args: List[str]):
        """Handle /ping command"""
        chat_id = message['chat']['id']
        self.send_notification("🏓 Pong! Bot is responsive.", chat_id)
    
    def _cmd_info(self, message: Dict, args: List[str]):
        """Handle /info command - show bot info"""
        chat_id = message['chat']['id']
        try:
            bot_info = self._get_bot_info()
            if bot_info:
                info_text = f"""
🤖 <b>Bot Information</b>

👤 Username: @{bot_info.get('username', 'unknown')}
🏷 Name: {bot_info.get('first_name', 'Unknown')}
🆔 ID: {bot_info.get('id', 'unknown')}
                """.strip()
            else:
                info_text = "❌ Could not retrieve bot information"
            
            self.send_notification(info_text, chat_id)
        except Exception as e:
            self.send_notification(f"❌ Error getting bot info: {e}", chat_id)
    
    def _get_bot_info(self) -> Dict:
        """Get bot information"""
        try:
            self._rate_limit()
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            response.raise_for_status()
            return response.json().get('result', {})
        except:
            return {}
    
    def test_connection(self) -> bool:
        """Test if the bot token is valid"""
        bot_info = self._get_bot_info()
        return bool(bot_info)

    def send_table(self, data: List[Dict], title: str = "", 
                   chat_id: Optional[int] = None, 
                   color_rules: Optional[Dict] = None,
                   max_rows: int = 20, atm_strike: int = 0) -> bool:
        """
        Send a formatted table as a Telegram message.
        
        Args:
            data: List of dictionaries representing table rows
            title: Optional title for the table
            chat_id: Specific chat ID (if None, sends to all authorized chats)
            color_rules: Dictionary defining color rules for values
            max_rows: Maximum number of rows to include
            atm_strike: ATM strike for options formatting
            
        Returns:
            bool: True if sent successfully
            
        Example:
            data = [
                {'Symbol': 'BANKNIFTY25JUL57000CE', 'Strike': 57000, 'OI': 1812265, 'OI_Change': -0.65},
                {'Symbol': 'BANKNIFTY25JUL57100CE', 'Strike': 57100, 'OI': 577430, 'OI_Change': +1.80}
            ]
            color_rules = {
                'OI_Change': {'positive': '🟢', 'negative': '🔴', 'zero': '⚪'}
            }
        """
        if not data:
            return False
        
        # Limit rows
        if len(data) > max_rows:
            data = data[:max_rows]
        
        # Format the table
        formatted_table = self._format_table_html(data, title, color_rules, atm_strike)
        
        # Send the formatted table
        return self.send_notification(formatted_table, chat_id, parse_mode='HTML')
    
    def _format_table_html(self, data: List[Dict], title: str = "", 
                          color_rules: Optional[Dict] = None, atm_strike: int = 0) -> str:
        """Format data as a mobile-friendly table for Telegram"""
        if not data:
            return "No data to display"
        
        # Get column headers
        headers = list(data[0].keys())
        
        # Build the table with better mobile formatting
        lines = []
        
        # Add title if provided
        if title:
            lines.append(f"<b>📊 {title}</b>\n")
        
        # Start monospace block
        lines.append("<code>")
        
        # For mobile-friendly format, use a more compact layout
        for i, row in enumerate(data):
            if i > 0:  # Add separator between rows
                lines.append("─" * 40)
            
            # Format each row in a compact way
            strike = row.get('Strike', 'N/A')
            symbol = self._format_symbol_short(row.get('Symbol', 'N/A'))
            oi = row.get('Latest_OI', 'N/A')
            
            # Main row with strike and OI - enhanced with ATM/ITM/OTM differentiation
            strike_text = self._format_cell_value(strike, 'Strike', color_rules, atm_strike)
            lines.append(f"{strike_text} | OI: {oi}")
            lines.append(f"   {symbol}")
            
            # OI Changes row with color indicators and high change highlighting
            changes = []
            for col in headers:
                if 'OI_' in col and col != 'OI_Time':
                    value = row.get(col)
                    if value is not None:
                        formatted_value = self._format_cell_value(value, col, color_rules, atm_strike)
                        period = col.replace('OI_', '').replace('m', 'min')
                        changes.append(f"{period}: {formatted_value}")
            
            if changes:
                lines.append(f"   {' | '.join(changes)}")
        
        lines.append("</code>")
        
        return "\n".join(lines)
    
    def _format_symbol_short(self, symbol: str) -> str:
        """Shorten symbol names for better mobile display"""
        if not symbol or symbol == 'N/A':
            return symbol
        
        # Extract key parts: BANKNIFTY25JUL57000CE -> BN 57000 CE
        if 'BANKNIFTY' in symbol:
            parts = symbol.replace('NSE:', '').replace('BANKNIFTY', 'BN')
            # Extract strike and option type
            if 'CE' in parts:
                strike_part = parts.split('CE')[0][-5:]  # Last 5 chars before CE
                return f"BN {strike_part} CE"
            elif 'PE' in parts:
                strike_part = parts.split('PE')[0][-5:]  # Last 5 chars before PE
                return f"BN {strike_part} PE"
        elif 'NIFTY' in symbol:
            parts = symbol.replace('NSE:', '').replace('NIFTY', 'NF')
            if 'CE' in parts:
                strike_part = parts.split('CE')[0][-5:]
                return f"NF {strike_part} CE"
            elif 'PE' in parts:
                strike_part = parts.split('PE')[0][-5:]
                return f"NF {strike_part} PE"
        
        # Fallback: just return first 15 chars
        return symbol[:15] + "..." if len(symbol) > 15 else symbol
    
    def send_formatted_notification(self, message: str, 
                                  notification_type: str = "info",
                                  chat_id: Optional[int] = None) -> bool:
        """
        Send a notification with predefined formatting styles.
        
        Args:
            message: The message content
            notification_type: Type of notification (info, success, warning, error, alert)
            chat_id: Specific chat ID
            
        Returns:
            bool: True if sent successfully
        """
        # Predefined styles
        styles = {
            'info': {'icon': 'ℹ️', 'color': 'blue'},
            'success': {'icon': '✅', 'color': 'green'},
            'warning': {'icon': '⚠️', 'color': 'orange'},
            'error': {'icon': '❌', 'color': 'red'},
            'alert': {'icon': '🚨', 'color': 'red'},
            'profit': {'icon': '💰', 'color': 'green'},
            'loss': {'icon': '📉', 'color': 'red'},
            'trade': {'icon': '📊', 'color': 'blue'}
        }
        
        style = styles.get(notification_type, styles['info'])
        
        formatted_message = f"{style['icon']} <b>{message}</b>"
        
        return self.send_notification(formatted_message, chat_id, parse_mode='HTML')

    def _format_cell_value(self, value: Any, column: str, color_rules: Optional[Dict], atm_strike: int = 0) -> str:
        """Format individual cell values with colors and indicators"""
        if value is None:
            return ""
        
        # Convert to string
        str_value = str(value)
        
        # Apply color rules if provided
        if color_rules and column in color_rules:
            rules = color_rules[column]
            
            # Handle Strike column with ATM/ITM/OTM differentiation
            if column == 'Strike' and isinstance(value, (int, float)):
                strike_value = float(value)
                if strike_value == atm_strike:
                    # ATM strike
                    indicator = rules.get('atm', '🎯')
                    return f"{indicator} <b>{str_value}</b>"  # Bold for ATM
                elif strike_value < atm_strike:
                    # ITM for calls, OTM for puts - use context-aware logic
                    indicator = rules.get('itm', '🟡')
                    return f"{indicator} {str_value}"
                else:
                    # OTM for calls, ITM for puts - use context-aware logic  
                    indicator = rules.get('otm', '🔵')
                    return f"{indicator} {str_value}"
            
            # Handle percentage values with high change highlighting
            elif isinstance(value, (int, float)):
                abs_value = abs(value)
                
                # Check for high changes (>±10%)
                if abs_value >= 10.0:
                    # High change - use bold and underline formatting
                    if value > 0:
                        indicator = rules.get('high_change', '⚡')
                        return f"{indicator} <b><u>+{str_value}</u></b>"
                    else:
                        indicator = rules.get('high_change', '⚡')
                        return f"{indicator} <b><u>{str_value}</u></b>"
                else:
                    # Regular formatting for normal changes
                    if value > 0:
                        indicator = rules.get('positive', '🟢')
                        return f"{indicator} +{str_value}"
                    elif value < 0:
                        indicator = rules.get('negative', '🔴')
                        return f"{indicator} {str_value}"
                    else:
                        indicator = rules.get('zero', '⚪')
                        return f"{indicator} {str_value}"
            
            # Handle string matching (legacy support)
            elif 'values' in rules:
                for pattern, indicator in rules['values'].items():
                    if pattern.lower() in str_value.lower():
                        return f"{indicator} {str_value}"
        
        return str_value
    
