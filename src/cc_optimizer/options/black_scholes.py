"""
Black-Scholes option pricing model.

Implements the Black-Scholes model for pricing European vanilla call and put options.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional

from .validators import validate_bs_parameters


class BlackScholesModel:
    """
    Black-Scholes option pricing model for European vanilla options.

    This class implements the classic Black-Scholes-Merton model for pricing
    European call and put options under the assumptions of:
    - Constant risk-free rate
    - Constant volatility (log-normal returns)
    - No dividends
    - Efficient markets (no arbitrage)
    - Continuous trading

    Attributes:
        S (float): Spot price of the underlying asset
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized, continuous compounding)
        sigma (float): Volatility of the underlying (annualized standard deviation)
        option_type (str): Type of option ('call' or 'put')

    Example:
        >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        >>> price = model.price()
        >>> print(f"Call option price: ${price:.2f}")
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ):
        """
        Initialize Black-Scholes model.

        Args:
            S: Spot price of the underlying asset (must be positive)
            K: Strike price (must be positive)
            T: Time to expiration in years (must be non-negative)
            r: Risk-free interest rate, annualized (e.g., 0.05 = 5%)
            sigma: Volatility, annualized (must be positive, e.g., 0.2 = 20%)
            option_type: Type of option ('call', 'put', 'c', or 'p')

        Raises:
            ValueError: If parameters are out of valid ranges
            TypeError: If parameters are not numeric

        Note:
            For crypto options, the risk-free rate is often set to 0 or a small
            value since traditional risk-free rate concepts may not apply.
        """
        # Validate and normalize parameters
        validated = validate_bs_parameters(S, K, T, r, sigma, option_type)

        self.S = validated['S']
        self.K = validated['K']
        self.T = validated['T']
        self.r = validated['r']
        self.sigma = validated['sigma']
        self.option_type = validated['option_type']

        # Cache d1 and d2 for efficiency (calculated lazily)
        self._d1_cached: Optional[float] = None
        self._d2_cached: Optional[float] = None
        self._cache_valid = False

    def _calculate_d_params(self) -> tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.

        These parameters are used in the cumulative normal distribution
        functions for option pricing.

        Returns:
            Tuple of (d1, d2)

        Notes:
            d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
            d2 = d1 - σ√T
        """
        if self._cache_valid:
            return self._d1_cached, self._d2_cached

        # Handle special case where T = 0 (option at expiration)
        if self.T == 0:
            # At expiration, option value is intrinsic value
            # Set d1 and d2 such that cumulative distribution gives correct result
            if self.S > self.K:
                # ITM call or OTM put
                self._d1_cached = np.inf
                self._d2_cached = np.inf
            elif self.S < self.K:
                # OTM call or ITM put
                self._d1_cached = -np.inf
                self._d2_cached = -np.inf
            else:
                # ATM at expiration
                self._d1_cached = 0
                self._d2_cached = 0
        else:
            # Standard Black-Scholes d1 and d2
            sqrt_T = np.sqrt(self.T)
            sigma_sqrt_T = self.sigma * sqrt_T

            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / sigma_sqrt_T
            d2 = d1 - sigma_sqrt_T

            self._d1_cached = d1
            self._d2_cached = d2

        self._cache_valid = True
        return self._d1_cached, self._d2_cached

    def _d1(self) -> float:
        """Get d1 parameter (cached)."""
        d1, _ = self._calculate_d_params()
        return d1

    def _d2(self) -> float:
        """Get d2 parameter (cached)."""
        _, d2 = self._calculate_d_params()
        return d2

    def price(self) -> float:
        """
        Calculate the option price using Black-Scholes formula.

        Returns:
            Option price (float)

        Notes:
            Call price: C = S·N(d1) - K·e^(-rT)·N(d2)
            Put price:  P = K·e^(-rT)·N(-d2) - S·N(-d1)

            Where N(·) is the cumulative standard normal distribution.

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> price = model.price()
            >>> print(f"${price:.2f}")
        """
        # Handle special case where T = 0 (option at expiration)
        if self.T == 0:
            if self.option_type == 'call':
                return max(self.S - self.K, 0)
            else:  # put
                return max(self.K - self.S, 0)

        d1, d2 = self._calculate_d_params()

        if self.option_type == 'call':
            # Call option price
            price = (
                self.S * norm.cdf(d1) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            )
        else:  # put
            # Put option price
            price = (
                self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) -
                self.S * norm.cdf(-d1)
            )

        return float(price)

    def intrinsic_value(self) -> float:
        """
        Calculate the intrinsic value of the option.

        Intrinsic value is the value if the option were exercised immediately.

        Returns:
            Intrinsic value (float, non-negative)

        Example:
            >>> model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> print(f"Intrinsic value: ${model.intrinsic_value():.2f}")
        """
        if self.option_type == 'call':
            return max(self.S - self.K, 0)
        else:  # put
            return max(self.K - self.S, 0)

    def time_value(self) -> float:
        """
        Calculate the time value of the option.

        Time value is the portion of the option price that exceeds intrinsic value.

        Returns:
            Time value (float, non-negative)

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> print(f"Time value: ${model.time_value():.2f}")
        """
        return self.price() - self.intrinsic_value()

    def moneyness(self) -> str:
        """
        Determine if the option is in-the-money, at-the-money, or out-of-the-money.

        Returns:
            'ITM', 'ATM', or 'OTM'

        Example:
            >>> model = BlackScholesModel(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> print(model.moneyness())  # 'ITM'
        """
        moneyness_ratio = self.S / self.K
        tolerance = 1e-6  # Consider ATM if within 0.0001% of strike

        if abs(moneyness_ratio - 1.0) < tolerance:
            return 'ATM'
        elif self.option_type == 'call':
            return 'ITM' if moneyness_ratio > 1.0 else 'OTM'
        else:  # put
            return 'ITM' if moneyness_ratio < 1.0 else 'OTM'

    @classmethod
    def from_iv_model(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        token: str,
        tenor_days: float,
        strike_percent: float = 100.0,
        option_type: str = 'call',
        iv_model: str = 'model_01',
        quote_type: str = 'mid'
    ) -> 'BlackScholesModel':
        """
        Create BlackScholesModel using implied volatility from hist_vol_model.

        This alternative constructor integrates with the hist_vol_model project
        to automatically fetch implied volatility for the given token and parameters.

        Args:
            S: Spot price of the underlying asset
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate
            token: Token symbol (e.g., 'BTC', 'ETH', 'HBAR')
            tenor_days: Option tenor in days (e.g., 7, 30, 90). Can be fractional.
            strike_percent: Strike as percentage of spot (e.g., 100.0 for ATM)
            option_type: 'call' or 'put'
            iv_model: IV model to use ('model_01', etc.)
            quote_type: Quote type ('bid', 'mid', 'ask')

        Returns:
            BlackScholesModel instance with IV from hist_vol_model

        Raises:
            ImportError: If hist_vol_model package is not installed
            ValueError: If IV cannot be computed

        Example:
            >>> model = BlackScholesModel.from_iv_model(
            ...     S=0.15, K=0.15, T=30/365, r=0.0,
            ...     token='HBAR', tenor_days=30.0, strike_percent=100.0,
            ...     option_type='call'
            ... )
            >>> price = model.price()

        Note:
            hist_vol_model must be installed as a package:
            pip install -e /path/to/hist_vol_model
        """
        # Import here to allow optional integration
        try:
            from .iv_helpers import compute_iv_with_cache
        except ImportError as e:
            raise ImportError(
                "hist_vol_model package not found. "
                "Please install it with: pip install -e /Users/neo/Velar/hist_vol_model"
            ) from e

        # Get IV from hist_vol_model (with caching for performance)
        try:
            sigma = compute_iv_with_cache(
                model=iv_model,
                token=token,
                tenor_days=tenor_days,
                strike_percent=strike_percent,
                quote_type=quote_type,
                use_cache=True
            )
        except Exception as e:
            raise ValueError(
                f"Failed to compute IV from hist_vol_model. "
                f"Parameters: token={token}, tenor_days={tenor_days}, "
                f"strike_percent={strike_percent}, quote_type={quote_type}. "
                f"Error: {e}"
            ) from e

        if sigma is None or np.isnan(sigma) or sigma <= 0:
            raise ValueError(
                f"Invalid IV returned from hist_vol_model: {sigma}. "
                f"Check parameters: token={token}, tenor_days={tenor_days}, "
                f"strike_percent={strike_percent}"
            )

        return cls(S, K, T, r, sigma, option_type)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"BlackScholesModel(S={self.S}, K={self.K}, T={self.T:.4f}, "
            f"r={self.r:.4f}, sigma={self.sigma:.4f}, option_type='{self.option_type}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        price = self.price()
        return (
            f"{self.option_type.upper()} Option:\n"
            f"  Spot: ${self.S:.2f}\n"
            f"  Strike: ${self.K:.2f}\n"
            f"  Time to expiry: {self.T:.4f} years\n"
            f"  Risk-free rate: {self.r:.2%}\n"
            f"  Volatility: {self.sigma:.2%}\n"
            f"  Price: ${price:.4f}\n"
            f"  Moneyness: {self.moneyness()}"
        )


def bs_pricer(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Simple functional interface for Black-Scholes option pricing.

    This function provides a quick way to price an option without creating
    a model object explicitly.

    Args:
        S: Spot price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price (float)

    Example:
        >>> price = bs_pricer(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        >>> print(f"Call option price: ${price:.2f}")
    """
    model = BlackScholesModel(S, K, T, r, sigma, option_type)
    return model.price()
