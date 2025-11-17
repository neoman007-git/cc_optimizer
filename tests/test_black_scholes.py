"""
Tests for Black-Scholes option pricing model.

Tests cover:
- Basic pricing functionality
- Put-call parity
- Boundary conditions
- Edge cases
- Validation against py_vollib (reference implementation)
"""

import pytest
import numpy as np
from cc_optimizer.options import BlackScholesModel, bs_pricer


class TestBlackScholesModel:
    """Test cases for BlackScholesModel class."""

    def test_basic_call_pricing(self):
        """Test basic call option pricing with known values."""
        # Standard ATM call
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        price = model.price()

        # Price should be positive and less than spot price
        assert price > 0
        assert price < 100

        # For ATM call with reasonable parameters, price should be in expected range
        # Rough estimate: ATM call ~ 0.4 * S * sigma * sqrt(T) = 0.4 * 100 * 0.2 * 1 = 8
        assert 5 < price < 15

    def test_basic_put_pricing(self):
        """Test basic put option pricing with known values."""
        # Standard ATM put
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        price = model.price()

        # Price should be positive
        assert price > 0

        # ATM put should be less than ATM call (with positive r)
        call_model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        call_price = call_model.price()
        assert price < call_price

    def test_put_call_parity(self):
        """
        Test put-call parity: C - P = S - K*exp(-rT).

        This fundamental relationship must hold for European options.
        """
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        call_model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
        put_model = BlackScholesModel(S, K, T, r, sigma, option_type='put')

        call_price = call_model.price()
        put_price = put_model.price()

        # C - P = S - K*exp(-rT)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 1e-10

    def test_put_call_parity_various_parameters(self):
        """Test put-call parity with various parameter combinations."""
        test_cases = [
            # (S, K, T, r, sigma)
            (100, 100, 1.0, 0.05, 0.2),  # ATM, 1 year
            (100, 90, 0.5, 0.03, 0.25),  # ITM call, 6 months
            (100, 110, 2.0, 0.02, 0.3),  # OTM call, 2 years
            (50, 50, 0.25, 0.01, 0.4),   # ATM, 3 months, high vol
            (200, 180, 0.1, 0.0, 0.5),   # Near expiry, r=0 (crypto-like)
        ]

        for S, K, T, r, sigma in test_cases:
            call_model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
            put_model = BlackScholesModel(S, K, T, r, sigma, option_type='put')

            call_price = call_model.price()
            put_price = put_model.price()

            lhs = call_price - put_price
            rhs = S - K * np.exp(-r * T)

            assert abs(lhs - rhs) < 1e-8, f"Put-call parity failed for {S, K, T, r, sigma}"

    def test_deep_itm_call(self):
        """Test deep in-the-money call option."""
        # Deep ITM: S >> K
        model = BlackScholesModel(S=150, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        price = model.price()

        # Deep ITM call should be approximately S - K*exp(-rT)
        expected = 150 - 100 * np.exp(-0.05 * 1)
        assert abs(price - expected) < 1.0  # Within $1

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option."""
        # Deep OTM: S << K
        model = BlackScholesModel(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        price = model.price()

        # Deep OTM call should be close to zero
        assert 0 < price < 1.0

    def test_deep_itm_put(self):
        """Test deep in-the-money put option."""
        # Deep ITM: S << K
        model = BlackScholesModel(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        price = model.price()

        # Deep ITM put should be approximately K*exp(-rT) - S
        expected = 100 * np.exp(-0.05 * 1) - 50
        assert abs(price - expected) < 1.0  # Within $1

    def test_zero_time_to_expiry_call(self):
        """Test call option at expiration (T=0)."""
        # ITM at expiration
        model = BlackScholesModel(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type='call')
        price = model.price()
        assert abs(price - 10.0) < 1e-10

        # OTM at expiration
        model = BlackScholesModel(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type='call')
        price = model.price()
        assert abs(price - 0.0) < 1e-10

        # ATM at expiration
        model = BlackScholesModel(S=100, K=100, T=0, r=0.05, sigma=0.2, option_type='call')
        price = model.price()
        assert abs(price - 0.0) < 1e-10

    def test_zero_time_to_expiry_put(self):
        """Test put option at expiration (T=0)."""
        # ITM at expiration
        model = BlackScholesModel(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type='put')
        price = model.price()
        assert abs(price - 10.0) < 1e-10

        # OTM at expiration
        model = BlackScholesModel(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type='put')
        price = model.price()
        assert abs(price - 0.0) < 1e-10

    def test_very_long_expiry(self):
        """Test option with very long time to expiry."""
        # 10-year option
        model = BlackScholesModel(S=100, K=100, T=10, r=0.05, sigma=0.2, option_type='call')
        price = model.price()

        # Long-dated ATM call should be substantial
        assert price > 20
        assert price < 100  # But still less than spot

    def test_high_volatility(self):
        """Test option with very high volatility."""
        # 100% annual volatility (crypto-like)
        model = BlackScholesModel(S=100, K=100, T=1, r=0.0, sigma=1.0, option_type='call')
        price = model.price()

        # High vol should increase option value significantly
        assert price > 30

    def test_low_volatility(self):
        """Test option with very low volatility."""
        # 5% annual volatility
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.05, option_type='call')
        price = model.price()

        # Low vol ATM option should be worth less
        assert 2 < price < 7

    def test_intrinsic_value_call(self):
        """Test intrinsic value calculation for call options."""
        # ITM call
        model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert abs(model.intrinsic_value() - 10.0) < 1e-10

        # OTM call
        model = BlackScholesModel(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert abs(model.intrinsic_value() - 0.0) < 1e-10

        # ATM call
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert abs(model.intrinsic_value() - 0.0) < 1e-10

    def test_intrinsic_value_put(self):
        """Test intrinsic value calculation for put options."""
        # ITM put
        model = BlackScholesModel(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert abs(model.intrinsic_value() - 10.0) < 1e-10

        # OTM put
        model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert abs(model.intrinsic_value() - 0.0) < 1e-10

    def test_time_value(self):
        """Test time value calculation."""
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        time_value = model.time_value()

        # ATM option should have positive time value
        assert time_value > 0

        # Time value should equal price for ATM option (zero intrinsic value)
        assert abs(time_value - model.price()) < 1e-10

    def test_moneyness_classification(self):
        """Test moneyness classification."""
        # ITM call
        model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert model.moneyness() == 'ITM'

        # OTM call
        model = BlackScholesModel(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert model.moneyness() == 'OTM'

        # ATM call
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert model.moneyness() == 'ATM'

        # ITM put
        model = BlackScholesModel(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert model.moneyness() == 'ITM'

        # OTM put
        model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert model.moneyness() == 'OTM'

    def test_option_type_normalization(self):
        """Test that various option type inputs are normalized correctly."""
        # Test 'call' variations
        for option_type in ['call', 'Call', 'CALL', 'c', 'C']:
            model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type=option_type)
            assert model.option_type == 'call'

        # Test 'put' variations
        for option_type in ['put', 'Put', 'PUT', 'p', 'P']:
            model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type=option_type)
            assert model.option_type == 'put'

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Negative spot price
        with pytest.raises(ValueError):
            BlackScholesModel(S=-100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

        # Negative strike price
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=-100, T=1, r=0.05, sigma=0.2, option_type='call')

        # Negative time to expiry
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=100, T=-1, r=0.05, sigma=0.2, option_type='call')

        # Negative volatility
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=-0.2, option_type='call')

        # Zero volatility
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0, option_type='call')

        # Invalid option type
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='invalid')

    def test_nan_inf_parameters(self):
        """Test that NaN and Inf parameters raise errors."""
        # NaN spot
        with pytest.raises(ValueError):
            BlackScholesModel(S=np.nan, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

        # Inf strike
        with pytest.raises(ValueError):
            BlackScholesModel(S=100, K=np.inf, T=1, r=0.05, sigma=0.2, option_type='call')

    def test_functional_api(self):
        """Test the functional bs_pricer interface."""
        # Should give same result as class-based interface
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        class_price = BlackScholesModel(S, K, T, r, sigma, option_type='call').price()
        func_price = bs_pricer(S, K, T, r, sigma, option_type='call')

        assert abs(class_price - func_price) < 1e-10

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

        # Should not raise exceptions
        str_repr = str(model)
        repr_str = repr(model)

        assert 'call' in str_repr.lower() or 'CALL' in str_repr
        assert 'BlackScholesModel' in repr_str


class TestBlackScholesVsPyVolib:
    """Test Black-Scholes implementation against py_vollib reference."""

    @pytest.fixture(autouse=True)
    def check_py_vollib(self):
        """Check if py_vollib is available, skip tests if not."""
        try:
            import py_vollib.black_scholes
            self.py_vollib_available = True
        except ImportError:
            self.py_vollib_available = False
            pytest.skip("py_vollib not installed")

    def test_call_price_vs_py_vollib(self):
        """Test call option pricing against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes as pyv

        test_cases = [
            # (S, K, T, r, sigma)
            (100, 100, 1.0, 0.05, 0.2),   # ATM, standard
            (100, 90, 0.5, 0.03, 0.25),   # ITM
            (100, 110, 2.0, 0.02, 0.3),   # OTM
            (50, 50, 0.25, 0.0, 0.5),     # Short-dated, high vol
            (200, 180, 0.1, 0.01, 0.15),  # Near expiry
        ]

        for S, K, T, r, sigma in test_cases:
            our_price = bs_pricer(S, K, T, r, sigma, option_type='call')
            ref_price = pyv.black_scholes('c', S, K, T, r, sigma)

            assert abs(our_price - ref_price) < 1e-8, \
                f"Price mismatch for {S, K, T, r, sigma}: {our_price} vs {ref_price}"

    def test_put_price_vs_py_vollib(self):
        """Test put option pricing against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes as pyv

        test_cases = [
            # (S, K, T, r, sigma)
            (100, 100, 1.0, 0.05, 0.2),   # ATM, standard
            (100, 110, 0.5, 0.03, 0.25),  # ITM
            (100, 90, 2.0, 0.02, 0.3),    # OTM
            (50, 50, 0.25, 0.0, 0.5),     # Short-dated, high vol
            (200, 220, 0.1, 0.01, 0.15),  # Near expiry
        ]

        for S, K, T, r, sigma in test_cases:
            our_price = bs_pricer(S, K, T, r, sigma, option_type='put')
            ref_price = pyv.black_scholes('p', S, K, T, r, sigma)

            assert abs(our_price - ref_price) < 1e-8, \
                f"Price mismatch for {S, K, T, r, sigma}: {our_price} vs {ref_price}"

    def test_edge_cases_vs_py_vollib(self):
        """Test edge cases against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes as pyv

        # Very short time to expiry
        S, K, T, r, sigma = 100, 100, 0.01, 0.05, 0.2
        our_call = bs_pricer(S, K, T, r, sigma, option_type='call')
        ref_call = pyv.black_scholes('c', S, K, T, r, sigma)
        assert abs(our_call - ref_call) < 1e-8

        # Very high volatility
        S, K, T, r, sigma = 100, 100, 1.0, 0.0, 2.0
        our_call = bs_pricer(S, K, T, r, sigma, option_type='call')
        ref_call = pyv.black_scholes('c', S, K, T, r, sigma)
        assert abs(our_call - ref_call) < 1e-6  # Slightly looser tolerance for extreme values

        # Zero interest rate (crypto-like)
        S, K, T, r, sigma = 50000, 50000, 30/365, 0.0, 0.8
        our_call = bs_pricer(S, K, T, r, sigma, option_type='call')
        ref_call = pyv.black_scholes('c', S, K, T, r, sigma)
        assert abs(our_call - ref_call) < 1e-6


class TestCryptoScenarios:
    """Test scenarios specific to cryptocurrency options."""

    def test_zero_interest_rate(self):
        """Test with zero interest rate (common for crypto)."""
        # BTC-like parameters
        model = BlackScholesModel(
            S=50000,
            K=50000,
            T=30/365,  # 30 days
            r=0.0,     # No risk-free rate in crypto
            sigma=0.8,  # 80% annual vol
            option_type='call'
        )
        price = model.price()

        # Should price successfully
        assert price > 0
        # High vol should give substantial premium
        assert price > 1000

    def test_high_volatility_scenario(self):
        """Test with high volatility typical of crypto."""
        # ETH-like parameters
        model = BlackScholesModel(
            S=3000,
            K=3000,
            T=7/365,   # 1 week
            r=0.0,
            sigma=1.2,  # 120% annual vol
            option_type='put'
        )
        price = model.price()

        # Should price successfully
        assert price > 0

    def test_short_dated_options(self):
        """Test very short-dated options (1 day)."""
        model = BlackScholesModel(
            S=100,
            K=100,
            T=1/365,  # 1 day
            r=0.0,
            sigma=0.5,
            option_type='call'
        )
        price = model.price()

        # Should have small but positive value
        assert 0 < price < 5
