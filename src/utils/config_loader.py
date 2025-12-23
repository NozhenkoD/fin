import os
import json
import yaml
from typing import Dict, List


class ConfigLoader:
    """
    Loads configuration files (YAML and JSON) for the stock screener.

    Handles loading rules, ticker metadata, and application settings.
    """

    @staticmethod
    def load_rules(path: str = "config/rules.yaml") -> dict:
        """
        Load screening rules from YAML or JSON file.

        Args:
            path: Path to rules file

        Returns:
            Dictionary with 'rules' and 'indicators' keys

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rules file not found: {path}")

        # Determine format from extension
        _, ext = os.path.splitext(path)

        with open(path, 'r') as f:
            if ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif ext == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .yaml, .yml, or .json")

        # Validate structure
        if 'rules' not in config:
            raise ValueError("Config file must contain 'rules' key")

        return config

    @staticmethod
    def load_tickers(path: str = "data/tickers.json") -> Dict[str, dict]:
        """
        Load ticker metadata from JSON file.

        Expected format:
        {
            "AAPL": {
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "sector": "Technology"
            },
            ...
        }

        Args:
            path: Path to tickers.json file

        Returns:
            Dictionary mapping ticker symbols to metadata

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tickers file not found: {path}\n"
                f"Please create a tickers.json file with your ticker list."
            )

        with open(path, 'r') as f:
            tickers = json.load(f)

        if not isinstance(tickers, dict):
            raise ValueError("Tickers file must contain a dictionary")

        return tickers

    @staticmethod
    def load_settings(path: str = "config/settings.yaml") -> dict:
        """
        Load application settings from YAML or JSON file.

        Args:
            path: Path to settings file

        Returns:
            Dictionary with settings

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(path):
            # Return default settings if file doesn't exist
            return {
                'data': {
                    'period': '1y',
                    'interval': '1d'
                },
                'display': {
                    'show_summary': True
                }
            }

        _, ext = os.path.splitext(path)

        with open(path, 'r') as f:
            if ext in ['.yaml', '.yml']:
                settings = yaml.safe_load(f)
            elif ext == '.json':
                settings = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        return settings

    @staticmethod
    def save_rules(config: dict, path: str = "config/rules.yaml"):
        """
        Save rules configuration to file.

        Args:
            config: Rules configuration dictionary
            path: Path to save file

        Raises:
            ValueError: If unsupported file format
        """
        _, ext = os.path.splitext(path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            if ext in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif ext == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def get_default_rules() -> dict:
        """
        Get default rules configuration.

        Returns:
            Dictionary with default rules and indicators
        """
        return {
            'rules': [
                {
                    'field': 'change',
                    'operator': '>',
                    'value': 2.0
                },
                {
                    'field': 'price',
                    'operator': '>',
                    'value': 'SMA200'
                },
                {
                    'field': 'market_cap',
                    'operator': '>',
                    'value': 2000000000  # 2B
                }
            ],
            'indicators': {
                'sma': [50, 200],
                'ema': [20],
                'avg_volume': {
                    'window': 14
                }
            }
        }
