"""
Helper utilities for hist_vol_model integration.

This module provides utility functions for working with the hist_vol_model package,
including checking availability, listing available tokens, and caching IV lookups.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional


def is_hist_vol_model_available() -> bool:
    """
    Check if hist_vol_model package is available.

    Returns:
        True if hist_vol_model is installed and can be imported, False otherwise

    Example:
        >>> if is_hist_vol_model_available():
        ...     from cc_optimizer.options import BlackScholesModel
        ...     model = BlackScholesModel.from_iv_model(...)
        ... else:
        ...     print("Please install hist_vol_model")
    """
    try:
        import hist_vol_model  # noqa: F401
        return True
    except ImportError:
        return False


def list_available_tokens(data_dir: Optional[Path] = None) -> List[str]:
    """
    List tokens available in hist_vol_model data directory.

    Args:
        data_dir: Optional custom data directory. If None, uses default hist_vol_model location.

    Returns:
        List of token symbols that have data available

    Raises:
        ImportError: If hist_vol_model is not installed
        FileNotFoundError: If data directory doesn't exist

    Example:
        >>> tokens = list_available_tokens()
        >>> print(f"Available tokens: {tokens}")
        Available tokens: ['ENA', 'HBAR']
    """
    if not is_hist_vol_model_available():
        raise ImportError(
            "hist_vol_model package not found. "
            "Please install it with: pip install -e /Users/neo/Velar/hist_vol_model"
        )

    # Determine data directory
    if data_dir is None:
        # Get default location from hist_vol_model package
        import hist_vol_model
        package_path = Path(hist_vol_model.__file__).parent
        data_dir = package_path.parent.parent / 'data' / 'raw'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all OHLCV parquet files
    # Pattern: BINANCE_{TOKEN}_OHLCV_1HRS_*.parquet
    tokens = set()
    for file_path in data_dir.glob("BINANCE_*_OHLCV_*.parquet"):
        # Extract token from filename
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            token = parts[1]  # Token is second part after BINANCE
            tokens.add(token)

    return sorted(list(tokens))


def get_hist_vol_model_info() -> dict:
    """
    Get information about the hist_vol_model installation.

    Returns:
        Dictionary with installation info including:
        - installed: bool
        - version: str or None
        - location: str or None
        - available_tokens: list or None

    Example:
        >>> info = get_hist_vol_model_info()
        >>> print(f"Installed: {info['installed']}")
        >>> print(f"Tokens: {info['available_tokens']}")
    """
    info = {
        'installed': False,
        'version': None,
        'location': None,
        'available_tokens': None
    }

    if not is_hist_vol_model_available():
        return info

    try:
        import hist_vol_model
        info['installed'] = True

        # Get version
        if hasattr(hist_vol_model, '__version__'):
            info['version'] = hist_vol_model.__version__
        else:
            info['version'] = 'unknown'

        # Get location
        info['location'] = str(Path(hist_vol_model.__file__).parent)

        # Get available tokens
        try:
            info['available_tokens'] = list_available_tokens()
        except Exception:
            info['available_tokens'] = None

    except Exception:
        pass

    return info


# Data-driven cache invalidation helpers
def _get_data_modification_time(token: str) -> float:
    """
    Get the last modification time of token's data file.

    This is used for cache invalidation - when data files are updated,
    the modification time changes and cache is automatically invalidated.

    Args:
        token: Token symbol (e.g., 'HBAR', 'ENA')

    Returns:
        Unix timestamp of last modification, or 0.0 if file not found
    """
    try:
        import hist_vol_model
        package_path = Path(hist_vol_model.__file__).parent
        data_dir = package_path.parent.parent / 'data' / 'raw'

        # Find token's data file(s)
        files = list(data_dir.glob(f"BINANCE_{token}_OHLCV_*.parquet"))
        if files:
            # Return latest modification time
            return max(f.stat().st_mtime for f in files)
        return 0.0
    except Exception:
        # If anything goes wrong, return 0 (no cache invalidation)
        return 0.0


def _get_config_modification_time(model: str) -> float:
    """
    Get the last modification time of model's config file.

    This is used for cache invalidation - when config files are updated,
    the modification time changes and cache is automatically invalidated.

    Args:
        model: Model name (e.g., 'model_01')

    Returns:
        Unix timestamp of last modification, or 0.0 if file not found
    """
    try:
        import hist_vol_model
        package_path = Path(hist_vol_model.__file__).parent
        config_dir = package_path.parent.parent / 'config'

        # Config file pattern: model_01_params.yaml
        config_file = config_dir / f"{model}_params.yaml"
        if config_file.exists():
            return config_file.stat().st_mtime
        return 0.0
    except Exception:
        return 0.0


# Enhanced cached IV computation with automatic invalidation
@lru_cache(maxsize=256)
def _compute_iv_cached(
    model: str,
    token: str,
    tenor_days: float,
    strike_percent: float,
    quote_type: str,
    data_mtime: float,
    config_mtime: float
) -> float:
    """
    Cached wrapper around hist_vol_model compute_iv with automatic invalidation.

    Uses LRU cache with data-driven invalidation. Cache is automatically
    invalidated when:
    - Data files are updated (new incremental data)
    - Config files are modified
    - New tokens are added (different data_mtime)

    Args:
        model: IV model to use ('model_01', etc.)
        token: Token symbol (e.g., 'HBAR', 'ENA')
        tenor_days: Option tenor in days (can be fractional)
        strike_percent: Strike as percentage of spot
        quote_type: Quote type ('bid', 'mid', 'ask')
        data_mtime: Data file modification timestamp (for cache invalidation)
        config_mtime: Config file modification timestamp (for cache invalidation)

    Returns:
        Implied volatility as decimal (e.g., 0.85 = 85%)

    Note:
        Cache automatically invalidates when underlying data or config changes.
        Cache size is 256 entries. Call clear_iv_cache() to manually clear.
    """
    from hist_vol_model.models.iv_api import compute_iv
    return compute_iv(model, token, tenor_days, strike_percent, quote_type)


def compute_iv_with_cache(
    model: str,
    token: str,
    tenor_days: float,
    strike_percent: float = 100.0,
    quote_type: str = 'mid',
    use_cache: bool = True
) -> float:
    """
    Compute IV with automatic caching and cache invalidation.

    This function provides intelligent caching that automatically invalidates
    when underlying data or configuration changes. Cache is invalidated when:
    - Data files are updated (new incremental data downloaded)
    - Config files are modified (parameters changed)
    - New tokens are added

    Args:
        model: IV model to use ('model_01', etc.)
        token: Token symbol (e.g., 'HBAR', 'ENA')
        tenor_days: Option tenor in days (can be fractional)
        strike_percent: Strike as percentage of spot (default 100.0 for ATM)
        quote_type: Quote type ('bid', 'mid', 'ask')
        use_cache: Whether to use caching (default True)

    Returns:
        Implied volatility as decimal (e.g., 0.85 = 85%)

    Example:
        >>> # First call computes IV (~1-2 seconds)
        >>> iv = compute_iv_with_cache('model_01', 'HBAR', 30.0, 100.0, 'mid')
        >>> # Second call returns cached result (microseconds)
        >>> iv = compute_iv_with_cache('model_01', 'HBAR', 30.0, 100.0, 'mid')
        >>>
        >>> # After data update, cache automatically invalidates
        >>> # download_new_data('HBAR')  # Data file mtime changes
        >>> iv = compute_iv_with_cache('model_01', 'HBAR', 30.0, 100.0, 'mid')
        >>> # Automatically recomputes with new data!

    Note:
        Cache invalidation is automatic and data-driven. You only need to
        manually call clear_iv_cache() if you want to force a full cache clear.
    """
    if use_cache:
        # Get modification times for cache invalidation
        data_mtime = _get_data_modification_time(token)
        config_mtime = _get_config_modification_time(model)

        # Cache key now includes modification times
        # If data or config changes, mtime changes, cache key changes, recomputes
        return _compute_iv_cached(
            model, token, tenor_days, strike_percent, quote_type,
            data_mtime, config_mtime
        )
    else:
        # Bypass cache entirely
        from hist_vol_model.models.iv_api import compute_iv
        return compute_iv(model, token, tenor_days, strike_percent, quote_type)


def clear_iv_cache() -> None:
    """
    Clear the IV computation cache.

    Call this if you want to force recomputation of IVs,
    for example after updating data in hist_vol_model.

    Example:
        >>> clear_iv_cache()  # Clear all cached IVs
        >>> cache_info = get_iv_cache_info()  # Check cache stats
    """
    _compute_iv_cached.cache_clear()


def get_iv_cache_info() -> dict:
    """
    Get cache statistics for IV computations.

    Returns:
        Dictionary with cache info:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - size: Current cache size
        - maxsize: Maximum cache size

    Example:
        >>> info = get_iv_cache_info()
        >>> print(f"Cache hit rate: {info['hits']/(info['hits']+info['misses']):.1%}")
    """
    cache_info = _compute_iv_cached.cache_info()
    return {
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'size': cache_info.currsize,
        'maxsize': cache_info.maxsize
    }
