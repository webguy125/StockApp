# Plugin System for StockApp
# Allows custom indicator development

import importlib
import os
from pathlib import Path

class PluginManager:
    """Manage custom indicator plugins"""

    def __init__(self, plugin_dir='plugins'):
        self.plugin_dir = Path(__file__).parent
        self.plugins = {}

    def discover_plugins(self):
        """Discover all available plugins"""
        for file in self.plugin_dir.glob('*.py'):
            if file.stem not in ['__init__', 'base_plugin']:
                try:
                    module = importlib.import_module(f'plugins.{file.stem}')
                    if hasattr(module, 'Plugin'):
                        self.plugins[file.stem] = module.Plugin()
                except Exception as e:
                    print(f"Failed to load plugin {file.stem}: {e}")
        return list(self.plugins.keys())

    def get_plugin(self, name):
        """Get a specific plugin"""
        return self.plugins.get(name)

    def execute_plugin(self, name, data, params=None):
        """Execute a plugin on data"""
        plugin = self.get_plugin(name)
        if plugin:
            return plugin.calculate(data, params or {})
        return None
