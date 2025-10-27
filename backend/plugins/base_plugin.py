# Base Plugin Class for Custom Indicators

class BasePlugin:
    """
    Base class for all custom indicator plugins.

    To create a custom indicator:
    1. Create a new .py file in the plugins directory
    2. Import this BasePlugin class
    3. Create a Plugin class that inherits from BasePlugin
    4. Implement the calculate() method

    Example:
    --------
    from plugins.base_plugin import BasePlugin

    class Plugin(BasePlugin):
        name = "My Custom Indicator"
        version = "1.0.0"
        description = "Calculates a custom technical indicator"

        def calculate(self, df, params=None):
            # Your calculation logic here
            period = params.get('period', 14)
            df['MyIndicator'] = df['Close'].rolling(period).mean()
            return df['MyIndicator'].tolist()
    """

    name = "Base Plugin"
    version = "0.0.0"
    description = "Base plugin class"
    author = "Unknown"
    parameters = {}

    def calculate(self, df, params=None):
        """
        Calculate the indicator

        Args:
            df: pandas DataFrame with OHLCV data
            params: dict of parameters

        Returns:
            list or dict of calculated values
        """
        raise NotImplementedError("Plugin must implement calculate() method")

    def validate_params(self, params):
        """Validate input parameters"""
        return True

    def get_info(self):
        """Get plugin information"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'parameters': self.parameters
        }
