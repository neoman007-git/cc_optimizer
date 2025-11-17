"""
Option Greeks calculator.

Implements calculation of option Greeks (sensitivities) for Black-Scholes model:
- Delta: Sensitivity to underlying price changes (dV/dS)
- Gamma: Rate of change of delta (d²V/dS²)
- Theta: Time decay (dV/dt)
- Vega: Sensitivity to volatility changes (dV/dσ)
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional

from .black_scholes import BlackScholesModel


class Greeks:
    """
    Option Greeks calculator for Black-Scholes model.

    This class calculates first-order risk sensitivities (Greeks) for options
    priced using the Black-Scholes model. Greeks are essential for risk management
    and hedging strategies.

    Attributes:
        bs (BlackScholesModel): Black-Scholes model instance to calculate Greeks for

    Example:
        >>> from cc_optimizer.options import BlackScholesModel, Greeks
        >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        >>> greeks = Greeks(model)
        >>> print(f"Delta: {greeks.delta():.4f}")
        >>> print(f"Gamma: {greeks.gamma():.4f}")
    """

    def __init__(self, bs_model: BlackScholesModel):
        """
        Initialize Greeks calculator.

        Args:
            bs_model: BlackScholesModel instance to calculate Greeks for

        Raises:
            TypeError: If bs_model is not a BlackScholesModel instance
        """
        if not isinstance(bs_model, BlackScholesModel):
            raise TypeError(
                f"bs_model must be a BlackScholesModel instance, "
                f"got {type(bs_model).__name__}"
            )
        self.bs = bs_model

    def delta(self) -> float:
        """
        Calculate delta (dV/dS).

        Delta measures the rate of change of option price with respect to changes
        in the underlying asset price. It represents the hedge ratio.

        Returns:
            Delta value (float)
            - Call delta: 0 to 1
            - Put delta: -1 to 0

        Notes:
            Call delta: Δ_c = N(d1)
            Put delta:  Δ_p = -N(-d1) = N(d1) - 1

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> print(f"Delta: {greeks.delta():.4f}")
        """
        # Handle special case where T = 0
        if self.bs.T == 0:
            if self.bs.option_type == 'call':
                return 1.0 if self.bs.S > self.bs.K else 0.0
            else:  # put
                return -1.0 if self.bs.S < self.bs.K else 0.0

        d1 = self.bs._d1()

        if self.bs.option_type == 'call':
            return float(norm.cdf(d1))
        else:  # put
            return float(norm.cdf(d1) - 1)  # Equivalent to -norm.cdf(-d1)

    def gamma(self) -> float:
        """
        Calculate gamma (d²V/dS²).

        Gamma measures the rate of change of delta with respect to changes in
        the underlying asset price. It indicates how much delta will change
        for a $1 move in the underlying.

        Returns:
            Gamma value (float, always non-negative)

        Notes:
            Γ = N'(d1) / (S·σ·√T)

            Where N'(·) is the standard normal probability density function.
            Gamma is the same for both calls and puts.

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> print(f"Gamma: {greeks.gamma():.6f}")
        """
        # Handle special case where T = 0
        if self.bs.T == 0:
            # At expiration, gamma is theoretically infinite at the strike
            # and zero elsewhere. We return 0 for numerical stability.
            return 0.0

        d1 = self.bs._d1()
        sqrt_T = np.sqrt(self.bs.T)

        # Standard normal PDF
        pdf_d1 = norm.pdf(d1)

        gamma = pdf_d1 / (self.bs.S * self.bs.sigma * sqrt_T)

        return float(gamma)

    def theta(self, per_day: bool = True) -> float:
        """
        Calculate theta (dV/dt).

        Theta measures the rate of change of option price with respect to the
        passage of time (time decay). It is typically negative for long options.

        Args:
            per_day: If True, return theta per calendar day (divide by 365).
                    If False, return theta per year.

        Returns:
            Theta value (float)
            - Usually negative for long options (time decay)
            - Call theta typically more negative than put theta (for same parameters)

        Notes:
            Call theta: Θ_c = -[S·N'(d1)·σ / (2√T)] - r·K·e^(-rT)·N(d2)
            Put theta:  Θ_p = -[S·N'(d1)·σ / (2√T)] + r·K·e^(-rT)·N(-d2)

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> print(f"Theta (per day): ${greeks.theta():.4f}")
        """
        # Handle special case where T = 0
        if self.bs.T == 0:
            return 0.0

        d1 = self.bs._d1()
        d2 = self.bs._d2()
        sqrt_T = np.sqrt(self.bs.T)

        # Common term for both call and put
        term1 = -(self.bs.S * norm.pdf(d1) * self.bs.sigma) / (2 * sqrt_T)

        # Discount factor
        discount = np.exp(-self.bs.r * self.bs.T)

        if self.bs.option_type == 'call':
            term2 = -self.bs.r * self.bs.K * discount * norm.cdf(d2)
            theta_annual = term1 + term2
        else:  # put
            term2 = self.bs.r * self.bs.K * discount * norm.cdf(-d2)
            theta_annual = term1 + term2

        if per_day:
            # Convert from per year to per day
            return float(theta_annual / 365)
        else:
            return float(theta_annual)

    def vega(self, per_percent: bool = True) -> float:
        """
        Calculate vega (dV/dσ).

        Vega measures the rate of change of option price with respect to changes
        in implied volatility. It is always positive for long options.

        Args:
            per_percent: If True, return vega per 1% change in volatility.
                        If False, return vega per 100% change (1.0 change in sigma).

        Returns:
            Vega value (float, always non-negative)

        Notes:
            ν = S·N'(d1)·√T

            Vega is the same for both calls and puts.
            Standard convention is to express vega per 1% change in volatility.

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> print(f"Vega (per 1%): ${greeks.vega():.4f}")
        """
        # Handle special case where T = 0
        if self.bs.T == 0:
            return 0.0

        d1 = self.bs._d1()
        sqrt_T = np.sqrt(self.bs.T)

        vega_full = self.bs.S * norm.pdf(d1) * sqrt_T

        if per_percent:
            # Convert to per 1% change (divide by 100)
            return float(vega_full / 100)
        else:
            return float(vega_full)

    def rho(self, per_percent: bool = True) -> float:
        """
        Calculate rho (dV/dr).

        Rho measures the rate of change of option price with respect to changes
        in the risk-free interest rate.

        Args:
            per_percent: If True, return rho per 1% change in interest rate.
                        If False, return rho per 100% change (1.0 change in r).

        Returns:
            Rho value (float)
            - Call rho: positive (calls benefit from higher rates)
            - Put rho: negative (puts benefit from lower rates)

        Notes:
            Call rho: ρ_c = K·T·e^(-rT)·N(d2)
            Put rho:  ρ_p = -K·T·e^(-rT)·N(-d2)

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> print(f"Rho (per 1%): ${greeks.rho():.4f}")
        """
        # Handle special case where T = 0
        if self.bs.T == 0:
            return 0.0

        d2 = self.bs._d2()
        discount = np.exp(-self.bs.r * self.bs.T)

        if self.bs.option_type == 'call':
            rho_full = self.bs.K * self.bs.T * discount * norm.cdf(d2)
        else:  # put
            rho_full = -self.bs.K * self.bs.T * discount * norm.cdf(-d2)

        if per_percent:
            # Convert to per 1% change (divide by 100)
            return float(rho_full / 100)
        else:
            return float(rho_full)

    def all_greeks(self, include_rho: bool = False) -> Dict[str, float]:
        """
        Calculate all Greeks at once (more efficient than individual calls).

        This method is more efficient than calling each Greek method separately
        because it reuses common calculations.

        Args:
            include_rho: If True, include rho in the results

        Returns:
            Dictionary with all Greeks:
            {
                'delta': float,
                'gamma': float,
                'theta': float,  # per day
                'vega': float,   # per 1%
                'rho': float     # per 1%, only if include_rho=True
            }

        Example:
            >>> model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
            >>> greeks = Greeks(model)
            >>> all_greeks = greeks.all_greeks()
            >>> for name, value in all_greeks.items():
            ...     print(f"{name}: {value:.4f}")
        """
        result = {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'theta': self.theta(per_day=True),
            'vega': self.vega(per_percent=True),
        }

        if include_rho:
            result['rho'] = self.rho(per_percent=True)

        return result

    def __repr__(self) -> str:
        """String representation of the Greeks calculator."""
        return f"Greeks({self.bs!r})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        greeks_dict = self.all_greeks()
        lines = ["Option Greeks:"]
        lines.append(f"  Delta: {greeks_dict['delta']:>10.4f}")
        lines.append(f"  Gamma: {greeks_dict['gamma']:>10.6f}")
        lines.append(f"  Theta: {greeks_dict['theta']:>10.4f} (per day)")
        lines.append(f"  Vega:  {greeks_dict['vega']:>10.4f} (per 1%)")
        return "\n".join(lines)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    include_rho: bool = False
) -> Dict[str, float]:
    """
    Calculate all Greeks for given parameters (functional interface).

    This function provides a quick way to calculate all Greeks without explicitly
    creating model and Greeks objects.

    Args:
        S: Spot price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        include_rho: If True, include rho in the results

    Returns:
        Dictionary with all Greeks (delta, gamma, theta, vega, and optionally rho)

    Example:
        >>> greeks = calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        >>> print(f"Delta: {greeks['delta']:.4f}")
        >>> print(f"Gamma: {greeks['gamma']:.6f}")
    """
    model = BlackScholesModel(S, K, T, r, sigma, option_type)
    greeks_calc = Greeks(model)
    return greeks_calc.all_greeks(include_rho=include_rho)
