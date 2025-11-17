"""
Tests for option Greeks calculations.

Tests cover:
- Delta, Gamma, Theta, Vega calculations
- Greek properties (bounds, signs, symmetries)
- Edge cases
- Validation against py_vollib (reference implementation)
"""

import pytest
import numpy as np
from cc_optimizer.options import BlackScholesModel, Greeks, calculate_greeks


class TestGreeks:
    """Test cases for Greeks class."""

    def test_call_delta_bounds(self):
        """Test that call delta is between 0 and 1."""
        test_cases = [
            # (S, K, T, r, sigma)
            (100, 100, 1.0, 0.05, 0.2),   # ATM
            (100, 90, 1.0, 0.05, 0.2),    # ITM
            (100, 110, 1.0, 0.05, 0.2),   # OTM
            (150, 100, 0.5, 0.0, 0.8),    # Deep ITM
            (50, 100, 0.5, 0.0, 0.5),     # Deep OTM
        ]

        for S, K, T, r, sigma in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
            greeks = Greeks(model)
            delta = greeks.delta()

            assert 0 <= delta <= 1, f"Call delta out of bounds: {delta} for {S, K, T, r, sigma}"

    def test_put_delta_bounds(self):
        """Test that put delta is between -1 and 0."""
        test_cases = [
            # (S, K, T, r, sigma)
            (100, 100, 1.0, 0.05, 0.2),   # ATM
            (100, 110, 1.0, 0.05, 0.2),   # ITM
            (100, 90, 1.0, 0.05, 0.2),    # OTM
            (50, 100, 0.5, 0.0, 0.8),     # Deep ITM
            (150, 100, 0.5, 0.0, 0.5),    # Deep OTM
        ]

        for S, K, T, r, sigma in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type='put')
            greeks = Greeks(model)
            delta = greeks.delta()

            assert -1 <= delta <= 0, f"Put delta out of bounds: {delta} for {S, K, T, r, sigma}"

    def test_delta_itm_otm_properties(self):
        """Test delta properties for ITM vs OTM options."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        # Deep ITM call delta should be close to 1
        itm_model = BlackScholesModel(S=150, K=100, T=T, r=r, sigma=sigma, option_type='call')
        itm_greeks = Greeks(itm_model)
        assert itm_greeks.delta() > 0.9

        # Deep OTM call delta should be close to 0
        otm_model = BlackScholesModel(S=50, K=100, T=T, r=r, sigma=sigma, option_type='call')
        otm_greeks = Greeks(otm_model)
        assert otm_greeks.delta() < 0.1

        # Deep ITM put delta should be close to -1
        itm_put = BlackScholesModel(S=50, K=100, T=T, r=r, sigma=sigma, option_type='put')
        itm_put_greeks = Greeks(itm_put)
        assert itm_put_greeks.delta() < -0.9

        # Deep OTM put delta should be close to 0
        otm_put = BlackScholesModel(S=150, K=100, T=T, r=r, sigma=sigma, option_type='put')
        otm_put_greeks = Greeks(otm_put)
        assert otm_put_greeks.delta() > -0.1

    def test_atm_call_delta_approx_half(self):
        """Test that ATM call delta is approximately 0.5."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)
        delta = greeks.delta()

        # ATM call delta should be around 0.5 (above 0.5 due to positive r and long T)
        # With r=0.05 and T=1, ATM delta is around 0.64
        assert 0.5 < delta < 0.7

    def test_put_call_delta_relationship(self):
        """Test that delta_put = delta_call - 1."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        call_model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
        put_model = BlackScholesModel(S, K, T, r, sigma, option_type='put')

        call_greeks = Greeks(call_model)
        put_greeks = Greeks(put_model)

        delta_call = call_greeks.delta()
        delta_put = put_greeks.delta()

        # delta_put = delta_call - 1
        assert abs(delta_put - (delta_call - 1)) < 1e-10

    def test_gamma_non_negative(self):
        """Test that gamma is always non-negative."""
        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),   # ATM call
            (100, 100, 1.0, 0.05, 0.2, 'put'),    # ATM put
            (100, 90, 1.0, 0.05, 0.2, 'call'),    # ITM call
            (100, 110, 1.0, 0.05, 0.2, 'put'),    # ITM put
            (150, 100, 0.5, 0.0, 0.8, 'call'),    # Deep ITM
            (50, 100, 0.5, 0.0, 0.5, 'put'),      # Deep ITM
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            gamma = greeks.gamma()

            assert gamma >= 0, f"Gamma is negative: {gamma} for {S, K, T, r, sigma, option_type}"

    def test_gamma_symmetry(self):
        """Test that gamma is the same for calls and puts."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        call_model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
        put_model = BlackScholesModel(S, K, T, r, sigma, option_type='put')

        call_greeks = Greeks(call_model)
        put_greeks = Greeks(put_model)

        gamma_call = call_greeks.gamma()
        gamma_put = put_greeks.gamma()

        # Gamma should be identical for calls and puts
        assert abs(gamma_call - gamma_put) < 1e-10

    def test_gamma_max_at_atm(self):
        """Test that gamma is maximum at ATM."""
        T, r, sigma = 1.0, 0.05, 0.2

        # ATM gamma
        atm_model = BlackScholesModel(S=100, K=100, T=T, r=r, sigma=sigma, option_type='call')
        atm_greeks = Greeks(atm_model)
        atm_gamma = atm_greeks.gamma()

        # ITM gamma (should be less)
        itm_model = BlackScholesModel(S=120, K=100, T=T, r=r, sigma=sigma, option_type='call')
        itm_greeks = Greeks(itm_model)
        itm_gamma = itm_greeks.gamma()

        # OTM gamma (should be less)
        otm_model = BlackScholesModel(S=80, K=100, T=T, r=r, sigma=sigma, option_type='call')
        otm_greeks = Greeks(otm_model)
        otm_gamma = otm_greeks.gamma()

        assert atm_gamma > itm_gamma
        assert atm_gamma > otm_gamma

    def test_theta_typically_negative(self):
        """Test that theta is typically negative for long options."""
        # Most long options (calls and puts) lose value over time
        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),   # ATM call
            (100, 100, 1.0, 0.05, 0.2, 'put'),    # ATM put
            (100, 90, 1.0, 0.05, 0.2, 'call'),    # ITM call
            (100, 110, 0.5, 0.05, 0.2, 'put'),    # ITM put
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            theta = greeks.theta(per_day=True)

            # Theta should typically be negative (time decay)
            assert theta <= 0, f"Theta is positive: {theta} for {S, K, T, r, sigma, option_type}"

    def test_theta_per_day_vs_per_year(self):
        """Test theta conversion from per year to per day."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        theta_per_day = greeks.theta(per_day=True)
        theta_per_year = greeks.theta(per_day=False)

        # Relationship: theta_per_day = theta_per_year / 365
        assert abs(theta_per_day * 365 - theta_per_year) < 1e-8

    def test_vega_non_negative(self):
        """Test that vega is always non-negative."""
        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),   # ATM call
            (100, 100, 1.0, 0.05, 0.2, 'put'),    # ATM put
            (100, 90, 1.0, 0.05, 0.2, 'call'),    # ITM call
            (100, 110, 1.0, 0.05, 0.2, 'put'),    # ITM put
            (150, 100, 0.5, 0.0, 0.8, 'call'),    # Deep ITM
            (50, 100, 0.5, 0.0, 0.5, 'put'),      # Deep ITM
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            vega = greeks.vega(per_percent=True)

            assert vega >= 0, f"Vega is negative: {vega} for {S, K, T, r, sigma, option_type}"

    def test_vega_symmetry(self):
        """Test that vega is the same for calls and puts."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        call_model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
        put_model = BlackScholesModel(S, K, T, r, sigma, option_type='put')

        call_greeks = Greeks(call_model)
        put_greeks = Greeks(put_model)

        vega_call = call_greeks.vega()
        vega_put = put_greeks.vega()

        # Vega should be identical for calls and puts
        assert abs(vega_call - vega_put) < 1e-10

    def test_vega_per_percent_vs_full(self):
        """Test vega conversion from full to per 1%."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        vega_per_percent = greeks.vega(per_percent=True)
        vega_full = greeks.vega(per_percent=False)

        # Relationship: vega_per_percent = vega_full / 100
        assert abs(vega_per_percent * 100 - vega_full) < 1e-8

    def test_vega_increases_with_time(self):
        """Test that vega increases with time to expiry."""
        S, K, r, sigma = 100, 100, 0.05, 0.2

        # Short-dated
        short_model = BlackScholesModel(S, K, T=0.1, r=r, sigma=sigma, option_type='call')
        short_greeks = Greeks(short_model)
        short_vega = short_greeks.vega()

        # Long-dated
        long_model = BlackScholesModel(S, K, T=2.0, r=r, sigma=sigma, option_type='call')
        long_greeks = Greeks(long_model)
        long_vega = long_greeks.vega()

        # Longer-dated options have higher vega
        assert long_vega > short_vega

    def test_rho_call_positive(self):
        """Test that rho is typically positive for calls."""
        # Calls benefit from higher interest rates
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)
        rho = greeks.rho()

        assert rho > 0

    def test_rho_put_negative(self):
        """Test that rho is typically negative for puts."""
        # Puts benefit from lower interest rates
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='put')
        greeks = Greeks(model)
        rho = greeks.rho()

        assert rho < 0

    def test_rho_per_percent_vs_full(self):
        """Test rho conversion from full to per 1%."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        rho_per_percent = greeks.rho(per_percent=True)
        rho_full = greeks.rho(per_percent=False)

        # Relationship: rho_per_percent = rho_full / 100
        assert abs(rho_per_percent * 100 - rho_full) < 1e-8

    def test_zero_time_greeks(self):
        """Test Greeks at expiration (T=0)."""
        # ITM call at expiration
        model = BlackScholesModel(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        # Delta should be 1 for ITM call
        assert abs(greeks.delta() - 1.0) < 1e-10

        # Gamma should be 0 at expiration
        assert abs(greeks.gamma() - 0.0) < 1e-10

        # Theta should be 0 at expiration
        assert abs(greeks.theta() - 0.0) < 1e-10

        # Vega should be 0 at expiration
        assert abs(greeks.vega() - 0.0) < 1e-10

    def test_all_greeks_method(self):
        """Test the all_greeks() convenience method."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        # Without rho
        all_greeks = greeks.all_greeks(include_rho=False)
        assert 'delta' in all_greeks
        assert 'gamma' in all_greeks
        assert 'theta' in all_greeks
        assert 'vega' in all_greeks
        assert 'rho' not in all_greeks

        # With rho
        all_greeks_with_rho = greeks.all_greeks(include_rho=True)
        assert 'delta' in all_greeks_with_rho
        assert 'gamma' in all_greeks_with_rho
        assert 'theta' in all_greeks_with_rho
        assert 'vega' in all_greeks_with_rho
        assert 'rho' in all_greeks_with_rho

        # Values should match individual method calls
        assert abs(all_greeks['delta'] - greeks.delta()) < 1e-10
        assert abs(all_greeks['gamma'] - greeks.gamma()) < 1e-10
        assert abs(all_greeks['theta'] - greeks.theta()) < 1e-10
        assert abs(all_greeks['vega'] - greeks.vega()) < 1e-10

    def test_functional_api(self):
        """Test the functional calculate_greeks interface."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        # Class-based
        model = BlackScholesModel(S, K, T, r, sigma, option_type='call')
        greeks_obj = Greeks(model)
        class_greeks = greeks_obj.all_greeks()

        # Functional
        func_greeks = calculate_greeks(S, K, T, r, sigma, option_type='call')

        # Should give same results
        assert abs(class_greeks['delta'] - func_greeks['delta']) < 1e-10
        assert abs(class_greeks['gamma'] - func_greeks['gamma']) < 1e-10
        assert abs(class_greeks['theta'] - func_greeks['theta']) < 1e-10
        assert abs(class_greeks['vega'] - func_greeks['vega']) < 1e-10

    def test_invalid_model_type(self):
        """Test that Greeks constructor rejects invalid model type."""
        with pytest.raises(TypeError):
            Greeks("not a model")

        with pytest.raises(TypeError):
            Greeks(123)

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        model = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
        greeks = Greeks(model)

        # Should not raise exceptions
        str_repr = str(greeks)
        repr_str = repr(greeks)

        assert 'Delta' in str_repr or 'delta' in str_repr
        assert 'Greeks' in repr_str


class TestGreeksVsPyVolib:
    """Test Greeks implementation against py_vollib reference."""

    @pytest.fixture(autouse=True)
    def check_py_vollib(self):
        """Check if py_vollib is available, skip tests if not."""
        try:
            import py_vollib.black_scholes.greeks.analytical
            self.py_vollib_available = True
        except ImportError:
            self.py_vollib_available = False
            pytest.skip("py_vollib not installed")

    def test_delta_vs_py_vollib(self):
        """Test delta calculation against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes.greeks.analytical as pyv_greeks

        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),
            (100, 100, 1.0, 0.05, 0.2, 'put'),
            (100, 90, 0.5, 0.03, 0.25, 'call'),
            (100, 110, 2.0, 0.02, 0.3, 'put'),
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            our_delta = greeks.delta()

            flag = 'c' if option_type == 'call' else 'p'
            ref_delta = pyv_greeks.delta(flag, S, K, T, r, sigma)

            assert abs(our_delta - ref_delta) < 1e-8, \
                f"Delta mismatch for {S, K, T, r, sigma, option_type}: {our_delta} vs {ref_delta}"

    def test_gamma_vs_py_vollib(self):
        """Test gamma calculation against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes.greeks.analytical as pyv_greeks

        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),
            (100, 100, 1.0, 0.05, 0.2, 'put'),
            (100, 90, 0.5, 0.03, 0.25, 'call'),
            (100, 110, 2.0, 0.02, 0.3, 'put'),
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            our_gamma = greeks.gamma()

            flag = 'c' if option_type == 'call' else 'p'
            ref_gamma = pyv_greeks.gamma(flag, S, K, T, r, sigma)

            assert abs(our_gamma - ref_gamma) < 1e-8, \
                f"Gamma mismatch for {S, K, T, r, sigma, option_type}: {our_gamma} vs {ref_gamma}"

    def test_theta_vs_py_vollib(self):
        """Test theta calculation against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes.greeks.analytical as pyv_greeks

        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),
            (100, 100, 1.0, 0.05, 0.2, 'put'),
            (100, 90, 0.5, 0.03, 0.25, 'call'),
            (100, 110, 2.0, 0.02, 0.3, 'put'),
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            our_theta = greeks.theta(per_day=True)  # py_vollib returns per day

            flag = 'c' if option_type == 'call' else 'p'
            ref_theta = pyv_greeks.theta(flag, S, K, T, r, sigma)

            assert abs(our_theta - ref_theta) < 1e-6, \
                f"Theta mismatch for {S, K, T, r, sigma, option_type}: {our_theta} vs {ref_theta}"

    def test_vega_vs_py_vollib(self):
        """Test vega calculation against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes.greeks.analytical as pyv_greeks

        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),
            (100, 100, 1.0, 0.05, 0.2, 'put'),
            (100, 90, 0.5, 0.03, 0.25, 'call'),
            (100, 110, 2.0, 0.02, 0.3, 'put'),
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            our_vega = greeks.vega(per_percent=True)  # py_vollib returns per 1%

            flag = 'c' if option_type == 'call' else 'p'
            ref_vega = pyv_greeks.vega(flag, S, K, T, r, sigma)

            assert abs(our_vega - ref_vega) < 1e-6, \
                f"Vega mismatch for {S, K, T, r, sigma, option_type}: {our_vega} vs {ref_vega}"

    def test_rho_vs_py_vollib(self):
        """Test rho calculation against py_vollib."""
        if not self.py_vollib_available:
            pytest.skip("py_vollib not installed")

        import py_vollib.black_scholes.greeks.analytical as pyv_greeks

        test_cases = [
            # (S, K, T, r, sigma, option_type)
            (100, 100, 1.0, 0.05, 0.2, 'call'),
            (100, 100, 1.0, 0.05, 0.2, 'put'),
            (100, 90, 0.5, 0.03, 0.25, 'call'),
            (100, 110, 2.0, 0.02, 0.3, 'put'),
        ]

        for S, K, T, r, sigma, option_type in test_cases:
            model = BlackScholesModel(S, K, T, r, sigma, option_type=option_type)
            greeks = Greeks(model)
            our_rho = greeks.rho(per_percent=True)  # py_vollib returns per 1%

            flag = 'c' if option_type == 'call' else 'p'
            ref_rho = pyv_greeks.rho(flag, S, K, T, r, sigma)

            assert abs(our_rho - ref_rho) < 1e-6, \
                f"Rho mismatch for {S, K, T, r, sigma, option_type}: {our_rho} vs {ref_rho}"


class TestCryptoGreeksScenarios:
    """Test Greeks in scenarios specific to cryptocurrency options."""

    def test_high_gamma_risk(self):
        """Test gamma risk in high volatility crypto environment."""
        # High vol, near ATM
        model = BlackScholesModel(
            S=50000,
            K=50000,
            T=7/365,  # 1 week
            r=0.0,
            sigma=1.0,  # 100% vol
            option_type='call'
        )
        greeks = Greeks(model)
        gamma = greeks.gamma()

        # High vol short-dated ATM options have significant gamma
        assert gamma > 0

    def test_vega_risk_crypto(self):
        """Test vega (vol risk) in crypto options."""
        # Typical crypto option
        model = BlackScholesModel(
            S=3000,
            K=3000,
            T=30/365,
            r=0.0,
            sigma=0.8,
            option_type='call'
        )
        greeks = Greeks(model)
        vega = greeks.vega()

        # Should have positive vega
        assert vega > 0

        # Vega should be meaningful (options are sensitive to vol changes)
        price = model.price()
        # 10% change in vol should have noticeable impact
        assert vega * 10 > price * 0.05  # >5% of price

    def test_theta_decay_short_dated(self):
        """Test theta (time decay) for short-dated crypto options."""
        # 1-day option
        model = BlackScholesModel(
            S=100,
            K=100,
            T=1/365,
            r=0.0,
            sigma=0.8,
            option_type='call'
        )
        greeks = Greeks(model)
        theta_per_day = greeks.theta(per_day=True)

        # Should have negative theta (time decay)
        assert theta_per_day < 0

        # For 1-day option, one day of theta should be substantial
        price = model.price()
        # One day decay should be significant portion of price
        assert abs(theta_per_day) > price * 0.1  # >10% of price per day
