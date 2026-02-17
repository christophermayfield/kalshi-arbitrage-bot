import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads():
    from src.utils.config import Config
    import tempfile
    import yaml

    config_data = {
        'kalshi': {
            'api_key_id': 'test-key',
            'base_url': 'https://test.api'
        },
        'trading': {
            'paper_mode': True,
            'min_profit_cents': 50
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = Config(config_path)

    assert config.get('kalshi.api_key_id') == 'test-key'
    assert config.paper_mode is True
    assert config.min_profit_cents == 50


def test_config_defaults():
    from src.utils.config import Config
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        config_path = f.name

    config = Config(config_path)

    assert config.paper_mode is True
    assert config.min_profit_cents == 10
    assert config.arbitrage_threshold == 0.99
