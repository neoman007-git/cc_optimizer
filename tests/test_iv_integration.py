"""
Integration tests for hist_vol_model connection.

Tests the integration between cc_optimizer option pricer and hist_vol_model,
including automatic IV fetching, quote types, and error handling.
"""

import pytest
import numpy as np

from cc_optimizer.options import (
    BlackScholesModel,
    Greeks,
    is_hist_vol_model_available,
    list_available_tokens,
    get_hist_vol_model_info
)


# Skip all tests if hist_vol_model is not available
pytestmark = pytest.mark.skipif(
    not is_hist_vol_model_available(),
    reason="hist_vol_model not installed"
)


class TestHistVolModelAvailability:
    """Test helper utilities for checking hist_vol_model availability."""

    def test_is_available(self):
        """Test that hist_vol_model is detected as available."""
        assert is_hist_vol_model_available() is True

    def test_list_tokens(self):
        """Test listing available tokens."""
        tokens = list_available_tokens()
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # HBAR should be available based on research
        assert 'HBAR' in tokens or 'ENA' in tokens

    def test_get_info(self):
        """Test getting hist_vol_model info."""
        info = get_hist_vol_model_info()
        assert info['installed'] is True
        assert info['version'] is not None
        assert info['location'] is not None
        assert info['available_tokens'] is not None
        assert len(info['available_tokens']) > 0


class TestFromIVModel:
    """Test the from_iv_model() class method integration."""

    def test_basic_pricing_hbar(self):
        """Test basic option pricing with HBAR auto-IV."""
        # Skip if HBAR not available
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        model = BlackScholesModel.from_iv_model(
            S=0.15,
            K=0.15,
            T=30/365,
            r=0.0,
            token='HBAR',
            tenor_days=30.0,
            strike_percent=100.0,
            option_type='call',
            quote_type='mid'
        )

        # Check model was created successfully
        assert isinstance(model, BlackScholesModel)
        assert model.S == 0.15
        assert model.K == 0.15
        assert model.r == 0.0
        assert model.option_type == 'call'

        # Check IV is reasonable (crypto typically 30-200%)
        assert 0.3 <= model.sigma <= 2.0, f"IV {model.sigma} outside reasonable range"

        # Check price is positive
        price = model.price()
        assert price > 0
        assert not np.isnan(price)
        assert not np.isinf(price)

    def test_basic_pricing_ena(self):
        """Test basic option pricing with ENA auto-IV."""
        tokens = list_available_tokens()
        if 'ENA' not in tokens:
            pytest.skip("ENA data not available")

        model = BlackScholesModel.from_iv_model(
            S=1.0,
            K=1.0,
            T=30/365,
            r=0.0,
            token='ENA',
            tenor_days=30.0,
            strike_percent=100.0,
            option_type='put',
            quote_type='mid'
        )

        assert isinstance(model, BlackScholesModel)
        assert model.option_type == 'put'
        assert 0.3 <= model.sigma <= 2.0
        assert model.price() > 0

    def test_quote_types_hbar(self):
        """Test different quote types return different IVs."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        # Get bid, mid, ask IVs
        bid_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            quote_type='bid'
        )

        mid_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            quote_type='mid'
        )

        ask_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            quote_type='ask'
        )

        # Bid IV should be <= Mid <= Ask
        assert bid_model.sigma <= mid_model.sigma <= ask_model.sigma
        assert bid_model.price() <= mid_model.price() <= ask_model.price()

    def test_strike_smile_adjustment(self):
        """Test that OTM strikes have different IV (smile adjustment)."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        # ATM
        atm_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            strike_percent=100.0
        )

        # 10% OTM call
        otm_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.165, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            strike_percent=110.0
        )

        # OTM should have higher IV due to smile
        assert otm_model.sigma >= atm_model.sigma

    def test_fractional_tenor(self):
        """Test that fractional tenors work (interpolation)."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        # 21.5 days (fractional, will be interpolated)
        model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=21.5/365, r=0.0,
            token='HBAR', tenor_days=21.5,
            strike_percent=100.0
        )

        assert isinstance(model, BlackScholesModel)
        assert 0.3 <= model.sigma <= 2.0
        assert model.price() > 0

    def test_greeks_with_auto_iv(self):
        """Test that Greeks calculations work with auto-IV."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0
        )

        greeks = Greeks(model)

        # Check all Greeks are computable
        delta = greeks.delta()
        gamma = greeks.gamma()
        theta = greeks.theta()
        vega = greeks.vega()
        rho = greeks.rho()

        # Sanity checks
        assert 0 <= delta <= 1  # ATM call delta
        assert gamma > 0
        assert theta < 0  # Options decay over time
        assert vega > 0
        assert not any(np.isnan([delta, gamma, theta, vega, rho]))

    def test_call_and_put_pricing(self):
        """Test both call and put options price correctly."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        call_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            option_type='call'
        )

        put_model = BlackScholesModel.from_iv_model(
            S=0.15, K=0.15, T=30/365, r=0.0,
            token='HBAR', tenor_days=30.0,
            option_type='put'
        )

        # ATM options with same IV should have similar prices
        call_price = call_model.price()
        put_price = put_model.price()

        assert call_price > 0
        assert put_price > 0
        # For ATM options with r=0, call and put should be very close
        assert abs(call_price - put_price) < 0.01


class TestErrorHandling:
    """Test error handling in integration."""

    def test_invalid_token(self):
        """Test that invalid token raises ValueError."""
        with pytest.raises(ValueError, match="Failed to compute IV"):
            BlackScholesModel.from_iv_model(
                S=100, K=100, T=30/365, r=0.0,
                token='INVALID_TOKEN_XYZ',
                tenor_days=30.0
            )

    def test_invalid_tenor(self):
        """Test that invalid tenor raises error."""
        tokens = list_available_tokens()
        if not tokens:
            pytest.skip("No tokens available")

        token = tokens[0]

        # Extreme tenor (likely outside available range)
        with pytest.raises(ValueError):
            BlackScholesModel.from_iv_model(
                S=100, K=100, T=1000/365, r=0.0,
                token=token,
                tenor_days=1000.0  # ~3 years, likely not available
            )

    def test_import_error_message(self):
        """Test that helpful error message is shown if package missing."""
        # This test can't easily simulate missing package, but we check the import path
        try:
            from hist_vol_model.models.iv_api import compute_iv
            # If import succeeds, check function exists
            assert callable(compute_iv)
        except ImportError as e:
            # If import fails, check error message is helpful
            assert "hist_vol_model" in str(e).lower()


class TestPerformance:
    """Test performance characteristics."""

    def test_multiple_pricings(self):
        """Test pricing multiple options (simulating backtest)."""
        tokens = list_available_tokens()
        if 'HBAR' not in tokens:
            pytest.skip("HBAR data not available")

        import time

        # Price 10 options with same parameters (should be fast)
        start = time.time()
        for _ in range(10):
            model = BlackScholesModel.from_iv_model(
                S=0.15, K=0.15, T=30/365, r=0.0,
                token='HBAR', tenor_days=30.0
            )
            _ = model.price()
        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds for 10 calls)
        # This is slow without caching, which we'll add next
        assert elapsed < 30, f"10 pricings took {elapsed:.2f}s (too slow)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
