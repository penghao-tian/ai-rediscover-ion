from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


import joblib

ArrayLike = Union[np.ndarray, Sequence[float]]


class VarianceContributionAnalyzer:
    """
    Class for analyzing physical variable contributions using Sobol variance decomposition.

    Main features
    --------
    1. Sobol/Saltelli global sampling
    2. First-order Sobol indices S1
    3. Total-effect Sobol indices ST
    4. Second-order interaction indices S2
    5. Bootstrap confidence intervals
    6. Single-variable ablation experiments
    7. Batch ablation experiments for multiple variables
    8. Significance assessment for weakly contributing variables
    9. Sobol index plots
    10. Log-scale plots of weak contributions
    11. Interaction heatmaps
    12. Ablation contribution plots
    13. Complete analysis reports and saved results

    Parameters
    ----------
    model : callable
        Physical formula or model.

        Expected input by default:
            X.shape = (n_samples, n_variables)

        Output:
            y.shape = (n_samples,)

        For example:
            def model(X):
                x1 = X[:, 0]
                x2 = X[:, 1]
                return x1 + x2**2

    variable_names : sequence of str
        Variable names, ordered to match the model input columns.

    bounds : sequence of (lower, upper)
        Value range for each variable.

    base_sample_size : int, default=4096
        Base sample size for Sobol analysis. Integer powers of 2 are recommended:
        1024、2048、4096、8192、16384。

    calc_second_order : bool, default=True
        Whether to compute second-order interactions.

    num_resamples : int, default=2000
        Number of resamples used by SALib for confidence interval estimation.

    conf_level : float, default=0.95
        Confidence level.

    random_seed : int, default=42
        Random seed.

    output_name : str, default="Model output"
        Model output name, used for plot labels.

    units : sequence of str or None
        Units for each variable; optional.

    vectorized : bool, default=True
        Whether the model can process the entire two-dimensional input matrix at once.
        If False, evaluate row by row.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], ArrayLike],
        variable_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        base_sample_size: int = 4096,
        calc_second_order: bool = True,
        num_resamples: int = 2000,
        conf_level: float = 0.95,
        random_seed: int = 42,
        output_name: str = "sailib_model_output",
        units: Optional[Sequence[str]] = None,
        vectorized: bool = True,
    ) -> None:

        self.model = model
        self.variable_names = list(variable_names)
        self.bounds = np.asarray(bounds, dtype=float)

        self.base_sample_size = int(base_sample_size)
        self.calc_second_order = bool(calc_second_order)
        self.num_resamples = int(num_resamples)
        self.conf_level = float(conf_level)
        self.random_seed = int(random_seed)
        self.output_name = output_name
        self.vectorized = bool(vectorized)

        if units is None:
            self.units = [""] * len(self.variable_names)
        else:
            self.units = list(units)

        self._validate_initial_inputs()

        self.problem = {
            "num_vars": len(self.variable_names),
            "names": self.variable_names,
            "bounds": self.bounds.tolist(),
        }

        # Sobol analysis results
        self.X_sobol: Optional[np.ndarray] = None
        self.Y_sobol: Optional[np.ndarray] = None
        self.sobol_result: Optional[dict] = None
        self.sensitivity_table: Optional[pd.DataFrame] = None
        self.second_order_table: Optional[pd.DataFrame] = None
        self.second_order_conf_table: Optional[pd.DataFrame] = None

        # Ablation analysis results
        self.X_ablation: Optional[np.ndarray] = None
        self.Y_ablation_full: Optional[np.ndarray] = None
        self.ablation_table: Optional[pd.DataFrame] = None
        self.ablation_details: dict[str, dict] = {}

    # ========================================================
    # Basic checks
    # ========================================================
    def _validate_initial_inputs(self) -> None:
        n_variables = len(self.variable_names)

        if n_variables == 0:
            raise ValueError("At least one variable must be provided.")

        if len(set(self.variable_names)) != n_variables:
            raise ValueError("variable_names contains duplicate names.")

        if self.bounds.shape != (n_variables, 2):
            raise ValueError(
                "The shape of bounds must be "
                f"({n_variables}, 2); the current shape is {self.bounds.shape}."
            )

        if not np.all(np.isfinite(self.bounds)):
            raise ValueError("bounds contains NaN or Inf.")

        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each variable must satisfy lower_bound < upper_bound.")

        if len(self.units) != n_variables:
            raise ValueError(
                "units must have the same length as variable_names."
            )

        if self.base_sample_size <= 0:
            raise ValueError("base_sample_size must be greater than zero.")

        if not 0 < self.conf_level < 1:
            raise ValueError("conf_level must be between 0 and 1.")

        if self.num_resamples <= 0:
            raise ValueError("num_resamples must be greater than zero.")

    def _evaluate_model(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the physical formula and validate the output."""

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Model input X must be a two-dimensional array.")

        if X.shape[1] != len(self.variable_names):
            raise ValueError(
                f"Model input must have {len(self.variable_names)} columns; "
                f"the current input has {X.shape[1]} columns."
            )

        if self.vectorized:
            Y = self.model(X)
        else:
            Y = np.array([self.model(row) for row in X])

        Y = np.asarray(Y, dtype=float).squeeze()

        if Y.ndim != 1:
            raise ValueError(
                "Model output must be a one-dimensional array."
                f"The current output shape is {Y.shape}."
            )

        if len(Y) != len(X):
            raise ValueError(
                "The model output length must match the number of input samples."
                f"Number of input samples={len(X)}, output length={len(Y)}."
            )

        if not np.all(np.isfinite(Y)):
            invalid_number = np.sum(~np.isfinite(Y))
            raise ValueError(
                f"Model output contains {invalid_number} NaN or Inf values."
                "Sobol sample rows cannot simply be deleted; check the formula or input ranges."
            )

        return Y

    def _require_sobol_results(self) -> None:
        if self.sensitivity_table is None:
            raise RuntimeError(
                "Sobol analysis has not been performed; call run_sobol() first."
            )

    def _require_ablation_results(self) -> None:
        if self.ablation_table is None:
            raise RuntimeError(
                "Ablation analysis has not been performed; call run_ablation_all() first."
            )

    # ========================================================
    # Sobol variance decomposition
    # ========================================================
    @property
    def expected_model_evaluations(self) -> int:
        """Return the number of model evaluations required for Sobol analysis."""

        d = len(self.variable_names)

        if self.calc_second_order:
            return self.base_sample_size * (2 * d + 2)

        return self.base_sample_size * (d + 2)

    def generate_sobol_samples(self) -> np.ndarray:
        """Generate a structured Sobol/Saltelli sampling matrix."""

        self.X_sobol = sobol_sample.sample(
            problem=self.problem,
            N=self.base_sample_size,
            calc_second_order=self.calc_second_order,
            scramble=True,
            seed=self.random_seed,
        )

        return self.X_sobol

    def run_sobol(self) -> pd.DataFrame:
        """
        Perform a complete Sobol variance decomposition.

        Returns
        -------
        pandas.DataFrame
            Include S1, ST, confidence intervals, and interaction contributions.
        """

        if self.X_sobol is None:
            self.generate_sobol_samples()

        self.Y_sobol = self._evaluate_model(self.X_sobol)

        output_variance = np.var(self.Y_sobol, ddof=1)

        if not np.isfinite(output_variance) or output_variance <= 0:
            raise ValueError(
                "Model output variance is zero or invalid; Sobol decomposition cannot be performed."
            )

        self.sobol_result = sobol_analyze.analyze(
            problem=self.problem,
            Y=self.Y_sobol,
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_resamples,
            conf_level=self.conf_level,
            print_to_console=False,
            seed=self.random_seed,
        )

        result = self.sobol_result

        table = pd.DataFrame({
            "Variable": self.variable_names,
            "Unit": self.units,
            "S1": result["S1"],
            "S1_conf": result["S1_conf"],
            "ST": result["ST"],
            "ST_conf": result["ST_conf"],
        })

        table["S1_lower"] = table["S1"] - table["S1_conf"]
        table["S1_upper"] = table["S1"] + table["S1_conf"]

        table["ST_lower"] = table["ST"] - table["ST_conf"]
        table["ST_upper"] = table["ST"] + table["ST_conf"]

        # The difference between total and first-order effects represents the contribution through interactions
        table["Interaction"] = table["ST"] - table["S1"]

        # Normalized total effects, used only for relative ranking among variables
        st_positive = np.maximum(table["ST"].to_numpy(), 0)
        st_sum = np.sum(st_positive)

        if st_sum > 0:
            table["Relative_ST"] = st_positive / st_sum
        else:
            table["Relative_ST"] = np.nan

        table["Rank_ST"] = (
            table["ST"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )

        table["S1_significant"] = table["S1_lower"] > 0
        table["ST_significant"] = table["ST_lower"] > 0

        table["Effect_type"] = table.apply(
            self._classify_effect_row,
            axis=1,
        )

        self.sensitivity_table = table.sort_values(
            "ST",
            ascending=False,
        ).reset_index(drop=True)

        if self.calc_second_order:
            self.second_order_table = pd.DataFrame(
                result["S2"],
                index=self.variable_names,
                columns=self.variable_names,
            )

            self.second_order_conf_table = pd.DataFrame(
                result["S2_conf"],
                index=self.variable_names,
                columns=self.variable_names,
            )

        return self.sensitivity_table.copy()

    @staticmethod
    def _classify_effect_row(row: pd.Series) -> str:
        """Classify variable effect types using S1, ST, and their confidence intervals."""

        if row["S1_lower"] > 0:
            if row["Interaction"] > row["S1"]:
                return "Independent and interaction-dominated"
            return "Stable independent effect"

        if row["S1_lower"] <= 0 and row["ST_lower"] > 0:
            return "Primarily interaction-driven"

        if row["ST"] > 0:
            return "Weak or uncertain effect"

        return "No resolved effect"

    # ========================================================
    # Retrieve and interpret single-variable results
    # ========================================================
    def get_variable_result(
        self,
        variable: Union[str, int],
    ) -> pd.Series:
        """Return the Sobol analysis results for the specified variable."""

        self._require_sobol_results()
        variable_name = self._resolve_variable_name(variable)

        row = self.sensitivity_table.loc[
            self.sensitivity_table["Variable"] == variable_name
        ]

        if row.empty:
            raise KeyError(f"Variable {variable_name!r} not found.")

        return row.iloc[0].copy()

    def interpret_variable(
        self,
        variable: Union[str, int],
        weak_threshold: float = 0.01,
    ) -> str:
        """
        Automatically interpret the contribution of a variable.

        weak_threshold
            Treat the contribution as weak when ST is below this value.
            For example, 0.01 represents approximately 1% of the output variance.
        """

        row = self.get_variable_result(variable)
        name = row["Variable"]

        statements = [
            f"Variable: {name}",
            f"S1 = {row['S1']:.6e}",
            (
                f"{self.conf_level:.0%} CI for S1: "
                f"[{row['S1_lower']:.6e}, "
                f"{row['S1_upper']:.6e}]"
            ),
            f"ST = {row['ST']:.6e}",
            (
                f"{self.conf_level:.0%} CI for ST: "
                f"[{row['ST_lower']:.6e}, "
                f"{row['ST_upper']:.6e}]"
            ),
            (
                "Estimated interaction contribution "
                f"(ST - S1) = {row['Interaction']:.6e}"
            ),
        ]

        if row["ST_lower"] > 0:
            statements.append(
                "The lower bound of the total-effect confidence interval is greater than zero, indicating that the overall contribution of this variable "
                "can be reliably identified at the current sample size."
            )
        else:
            statements.append(
                "The total-effect confidence interval includes zero; at the current sample size, this variable "
                "cannot be clearly distinguished from having zero contribution."
            )

        if row["S1_lower"] > 0:
            statements.append(
                "This variable has an identifiable independent contribution to variance."
            )
        elif row["ST_lower"] > 0:
            statements.append(
                "The first-order effect of this variable is weak, but its total effect is significant, indicating that its influence "
                "arises primarily from interactions with other variables."
            )

        if row["ST"] < weak_threshold:
            statements.append(
                f"ST is below {weak_threshold:g}, indicating a variable with a "
                "weak contribution in the global sense."
            )
        else:
            statements.append(
                f"ST is not below {weak_threshold:g}; its contribution should not simply be ignored."
            )

        if abs(row["Interaction"]) > abs(row["S1"]):
            statements.append(
                "The interaction contribution exceeds the independent contribution, so fixing this variable at its mean may "
                "substantially underestimate its physical influence."
            )

        return "\n".join(statements)

    def _resolve_variable_name(
        self,
        variable: Union[str, int],
    ) -> str:

        if isinstance(variable, str):
            if variable not in self.variable_names:
                raise KeyError(
                    f"{variable!r} is not in the variable list: "
                    f"{self.variable_names}"
                )
            return variable

        if isinstance(variable, (int, np.integer)):
            if not 0 <= variable < len(self.variable_names):
                raise IndexError("Variable index is out of range.")
            return self.variable_names[int(variable)]

        raise TypeError("variable must be a variable name or an integer index.")

    def _resolve_variable_index(
        self,
        variable: Union[str, int],
    ) -> int:

        variable_name = self._resolve_variable_name(variable)
        return self.variable_names.index(variable_name)

    # ========================================================
    # Ablation analysis
    # ========================================================
    def generate_ablation_samples(
        self,
        sample_size: int = 100000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Independently generate uniform random samples for ablation experiments.

        Saltelli structured samples are not used directly here, allowing ablation analysis to serve as
        an independent paired Monte Carlo experiment.
        """

        if sample_size <= 1:
            raise ValueError("sample_size must be greater than 1.")

        if seed is None:
            seed = self.random_seed + 1000

        rng = np.random.default_rng(seed)

        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        self.X_ablation = rng.uniform(
            low=lower,
            high=upper,
            size=(sample_size, len(self.variable_names)),
        )

        self.Y_ablation_full = self._evaluate_model(
            self.X_ablation
        )

        return self.X_ablation

    def run_ablation(
        self,
        variable: Union[str, int],
        fixed_value: Optional[float] = None,
        sample_size: int = 100000,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Perform an ablation experiment for a single variable.

        Fix the target variable, then compare the full and ablated outputs.

        Parameters
        ----------
        variable
            Variable name or variable index.

        fixed_value
            Fixed value. Defaults to the midpoint of the variable range.
            For highly skewed physical quantities, explicitly provide the median or a physical reference value.

        sample_size
            Monte Carlo sample size for the ablation experiment.
        """

        index = self._resolve_variable_index(variable)
        variable_name = self.variable_names[index]

        if (
            self.X_ablation is None
            or len(self.X_ablation) != sample_size
        ):
            self.generate_ablation_samples(
                sample_size=sample_size,
                seed=seed,
            )

        if fixed_value is None:
            fixed_value = np.mean(self.bounds[index])

        lower, upper = self.bounds[index]

        if not lower <= fixed_value <= upper:
            raise ValueError(
                f"The fixed value for {variable_name}, {fixed_value}, "
                f"is outside the range [{lower}, {upper}]."
            )

        X_fixed = self.X_ablation.copy()
        X_fixed[:, index] = fixed_value

        y_full = self.Y_ablation_full
        y_fixed = self._evaluate_model(X_fixed)
        delta = y_full - y_fixed

        full_variance = np.var(y_full, ddof=1)
        fixed_variance = np.var(y_fixed, ddof=1)
        delta_variance = np.var(delta, ddof=1)

        covariance = np.cov(y_full, y_fixed, ddof=1)[0, 1]

        if full_variance > 0:
            delta_variance_ratio = delta_variance / full_variance
            variance_change_ratio = (
                full_variance - fixed_variance
            ) / full_variance
        else:
            delta_variance_ratio = np.nan
            variance_change_ratio = np.nan

        if np.std(y_full) > 0 and np.std(y_fixed) > 0:
            correlation = np.corrcoef(y_full, y_fixed)[0, 1]
        else:
            correlation = np.nan

        result = {
            "Variable": variable_name,
            "Variable_index": index,
            "Fixed_value": float(fixed_value),
            "Full_mean": float(np.mean(y_full)),
            "Fixed_mean": float(np.mean(y_fixed)),
            "Mean_change": float(np.mean(delta)),
            "Mean_absolute_change": float(np.mean(np.abs(delta))),
            "Median_absolute_change": float(
                np.median(np.abs(delta))
            ),
            "RMS_change": float(
                np.sqrt(np.mean(delta**2))
            ),
            "Maximum_absolute_change": float(
                np.max(np.abs(delta))
            ),
            "Full_variance": float(full_variance),
            "Fixed_variance": float(fixed_variance),
            "Variance_change": float(
                full_variance - fixed_variance
            ),
            "Variance_change_ratio": float(
                variance_change_ratio
            ),
            "Delta_variance": float(delta_variance),
            "Delta_variance_ratio": float(
                delta_variance_ratio
            ),
            "Full_fixed_covariance": float(covariance),
            "Full_fixed_correlation": float(correlation),
            "Delta_q05": float(np.quantile(delta, 0.05)),
            "Delta_q25": float(np.quantile(delta, 0.25)),
            "Delta_q50": float(np.quantile(delta, 0.50)),
            "Delta_q75": float(np.quantile(delta, 0.75)),
            "Delta_q95": float(np.quantile(delta, 0.95)),
            "y_full": y_full,
            "y_fixed": y_fixed,
            "delta": delta,
        }

        self.ablation_details[variable_name] = result
        return result

    def run_ablation_all(
        self,
        fixed_values: Optional[dict[str, float]] = None,
        sample_size: int = 100000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform ablation experiments for all variables.

        Example of fixed_values
        The specific variable names depend on the input
        ----------------
        {
            "bz": 0,
            "r": 10,
            "sini",
            "cosi",
            "sind",
            "cosd",
            "dudz",
            "dvdz"
        }

        Unspecified variables are fixed at the midpoints of their ranges by default.
        """

        if fixed_values is None:
            fixed_values = {}

        self.generate_ablation_samples(
            sample_size=sample_size,
            seed=seed,
        )

        rows = []

        for variable_name in self.variable_names:
            result = self.run_ablation(
                variable=variable_name,
                fixed_value=fixed_values.get(variable_name),
                sample_size=sample_size,
                seed=seed,
            )

            rows.append({
                key: value
                for key, value in result.items()
                if key not in {"y_full", "y_fixed", "delta"}
            })

        table = pd.DataFrame(rows)

        table["Ablation_rank"] = (
            table["Delta_variance_ratio"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )

        self.ablation_table = table.sort_values(
            "Delta_variance_ratio",
            ascending=False,
        ).reset_index(drop=True)

        return self.ablation_table.copy()

    # ========================================================
    # Interactions
    # ========================================================
    def get_interactions_for_variable(
        self,
        variable: Union[str, int],
        sort: bool = True,
    ) -> pd.DataFrame:
        """Return second-order Sobol interactions between the specified variable and the other variables."""

        self._require_sobol_results()

        if not self.calc_second_order:
            raise RuntimeError(
                "calc_second_order=False; second-order interactions were not computed."
            )

        variable_name = self._resolve_variable_name(variable)
        index = self.variable_names.index(variable_name)

        rows = []

        for j, other_name in enumerate(self.variable_names):
            if j == index:
                continue

            i_low = min(index, j)
            i_high = max(index, j)

            s2 = self.sobol_result["S2"][i_low, i_high]
            s2_conf = self.sobol_result["S2_conf"][
                i_low, i_high
            ]

            rows.append({
                "Variable": variable_name,
                "Interacting_variable": other_name,
                "S2": s2,
                "S2_conf": s2_conf,
                "S2_lower": s2 - s2_conf,
                "S2_upper": s2 + s2_conf,
                "S2_significant": (s2 - s2_conf) > 0,
            })

        table = pd.DataFrame(rows)

        if sort:
            table = table.sort_values(
                "S2",
                ascending=False,
            )

        return table.reset_index(drop=True)

    # ========================================================
    # Common plot settings
    # ========================================================
    @staticmethod
    def _format_axis(ax: plt.Axes) -> None:
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _ordered_sensitivity_table(
        self,
        sort_by: str = "ST",
    ) -> pd.DataFrame:

        self._require_sobol_results()

        if sort_by not in self.sensitivity_table.columns:
            raise KeyError(f"Column {sort_by!r} not found.")

        return self.sensitivity_table.sort_values(
            sort_by,
            ascending=False,
        ).reset_index(drop=True)

    # ========================================================
    # Figure 1: S1 and ST with confidence intervals
    # ========================================================
    def plot_sobol_indices(
        self,
        sort_by: str = "ST",
        figsize: tuple[float, float] = (7.2, 4.6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        table = self._ordered_sensitivity_table(sort_by)
        x = np.arange(len(table))

        fig, ax = plt.subplots(figsize=figsize)

        ax.errorbar(
            x - 0.12,
            table["S1"],
            yerr=table["S1_conf"],
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1,
            label="First-order index, $S_i$",
        )

        ax.errorbar(
            x + 0.12,
            table["ST"],
            yerr=table["ST_conf"],
            fmt="s",
            markersize=5,
            capsize=3,
            linewidth=1,
            label="Total-order index, $S_{T_i}$",
        )

        ax.axhline(0, linewidth=0.8, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel("Physical variable")
        ax.set_ylabel("Sobol sensitivity index")
        ax.legend(frameon=False)

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 2: Total variable contributions
    # ========================================================
    def plot_variable_contributions(
        self,
        contribution: str = "ST",
        percentage: bool = True,
        sort: bool = True,
        figsize: tuple[float, float] = (7.2, 4.6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a bar chart of the contribution of each variable.

        contribution
            "S1": Independent contribution
            "ST": Total contribution
            "Interaction": Interaction contribution ST-S1
            "Relative_ST": Relative contribution normalized by the sum of ST
        """

        self._require_sobol_results()

        allowed = {"S1", "ST", "Interaction", "Relative_ST"}

        if contribution not in allowed:
            raise ValueError(
                f"contribution must be one of {allowed}."
            )

        table = self.sensitivity_table.copy()

        if sort:
            table = table.sort_values(
                contribution,
                ascending=False,
            )

        values = table[contribution].to_numpy(dtype=float)

        if percentage:
            values = values * 100
            ylabel = "Contribution (%)"
        else:
            ylabel = "Sensitivity index"

        x = np.arange(len(table))

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            x,
            values,
            width=0.68,
        )

        ax.axhline(0, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel("Physical variable")
        ax.set_ylabel(ylabel)

        for bar, value in zip(bars, values):
            vertical_alignment = (
                "bottom" if value >= 0 else "top"
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2g}",
                ha="center",
                va=vertical_alignment,
                fontsize=8,
            )

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 3: Stacked plot of independent and interaction contributions
    # ========================================================
    def plot_contribution_decomposition(
        self,
        percentage: bool = True,
        figsize: tuple[float, float] = (7.2, 4.6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        table = self._ordered_sensitivity_table("ST")
        
        print(type(table))
        table.to_csv("./salib_analysis/contribution.csv")
        
        
        independent = np.maximum(
            table["S1"].to_numpy(),
            0,
        )

        interaction = np.maximum(
            table["Interaction"].to_numpy(),
            0,
        )

        if percentage:
            independent *= 100
            interaction *= 100
            ylabel = "Variance contribution (%)"
        else:
            ylabel = "Sobol sensitivity index"

        x = np.arange(len(table))

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            x,
            independent,
            width=0.68,
            label="Independent contribution",
        )

        ax.bar(
            x,
            interaction,
            width=0.68,
            bottom=independent,
            label="Interaction contribution",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel("Physical variable")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 4: Weak contributions on a logarithmic scale
    # ========================================================
    def plot_weak_contributions(
        self,
        epsilon: float = 1e-8,
        figsize: tuple[float, float] = (7.2, 4.6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        table = self._ordered_sensitivity_table("ST")

        s1 = np.maximum(
            table["S1"].to_numpy(dtype=float),
            epsilon,
        )

        st = np.maximum(
            table["ST"].to_numpy(dtype=float),
            epsilon,
        )

        x = np.arange(len(table))
        width = 0.34

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            x - width / 2,
            s1,
            width=width,
            label="$S_i$",
        )

        ax.bar(
            x + width / 2,
            st,
            width=width,
            label="$S_{T_i}$",
        )

        ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel("Physical variable")
        ax.set_ylabel("Sobol sensitivity index")
        ax.legend(frameon=False)

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 5: Second-order interaction heatmap
    # ========================================================
    def plot_interaction_matrix(
        self,
        mask_lower_triangle: bool = True,
        figsize: tuple[float, float] = (6.2, 5.2),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        self._require_sobol_results()

        if not self.calc_second_order:
            raise RuntimeError(
                "Second-order effects were not computed; the interaction matrix cannot be plotted."
            )

        matrix = np.asarray(
            self.second_order_table,
            dtype=float,
        ).copy()

        # SALib stores S2 primarily in the upper triangle
        if mask_lower_triangle:
            matrix[np.tril_indices_from(matrix)] = np.nan

        fig, ax = plt.subplots(figsize=figsize)

        image = ax.imshow(
            matrix,
            origin="upper",
            aspect="equal",
        )

        ticks = np.arange(len(self.variable_names))

        ax.set_xticks(ticks)
        ax.set_xticklabels(
            self.variable_names,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(ticks)
        ax.set_yticklabels(self.variable_names)

        ax.set_xlabel("Interacting variable")
        ax.set_ylabel("Physical variable")

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Second-order Sobol index, $S_{ij}$")

        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 6: Interaction contributions for a specified variable
    # ========================================================
    def plot_variable_interactions(
        self,
        variable: Union[str, int],
        figsize: tuple[float, float] = (6.8, 4.5),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        table = self.get_interactions_for_variable(variable)
        variable_name = self._resolve_variable_name(variable)

        x = np.arange(len(table))

        fig, ax = plt.subplots(figsize=figsize)

        ax.errorbar(
            x,
            table["S2"],
            yerr=table["S2_conf"],
            fmt="o",
            capsize=3,
            linewidth=1,
        )

        ax.axhline(0, linewidth=0.8, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Interacting_variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel(
            f"Variable interacting with {variable_name}"
        )
        ax.set_ylabel("Second-order Sobol index, $S_{ij}$")

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 7: Ablation contributions for all variables
    # ========================================================
    def plot_ablation_contributions(
        self,
        metric: str = "Delta_variance_ratio",
        percentage: bool = True,
        figsize: tuple[float, float] = (7.2, 4.6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        self._require_ablation_results()

        if metric not in self.ablation_table.columns:
            raise KeyError(
                f"Metric {metric!r} does not exist in the ablation results."
            )

        table = self.ablation_table.sort_values(
            metric,
            ascending=False,
        )

        values = table[metric].to_numpy(dtype=float)

        if percentage:
            values *= 100
            ylabel = f"{metric} (%)"
        else:
            ylabel = metric

        x = np.arange(len(table))

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            x,
            values,
            width=0.68,
        )

        ax.axhline(0, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            table["Variable"],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel("Ablated physical variable")
        ax.set_ylabel(ylabel)

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 8: Comparison of full and ablated outputs
    # ========================================================
    def plot_ablation_comparison(
        self,
        variable: Union[str, int],
        bins: int = 80,
        figsize: tuple[float, float] = (7.0, 4.5),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        variable_name = self._resolve_variable_name(variable)

        if variable_name not in self.ablation_details:
            self.run_ablation(variable_name)

        result = self.ablation_details[variable_name]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            result["y_full"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="Full formula",
        )

        ax.hist(
            result["y_fixed"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"{variable_name} fixed",
        )

        ax.set_xlabel(self.output_name)
        ax.set_ylabel("Probability density")
        ax.legend(frameon=False)
        
        print(result["y_full"].shape)
        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Figure 9: Output differences caused by ablation
    # ========================================================
    def plot_ablation_delta(
        self,
        variable: Union[str, int],
        bins: int = 80,
        log_y: bool = False,
        figsize: tuple[float, float] = (7.0, 4.5),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:

        variable_name = self._resolve_variable_name(variable)

        if variable_name not in self.ablation_details:
            self.run_ablation(variable_name)

        delta = self.ablation_details[variable_name]["delta"]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            delta,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.65,
        )

        ax.axvline(
            0,
            linewidth=0.8,
            linestyle="--",
        )

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel(
            rf"$Y_{{\mathrm{{full}}}}"
            rf"-Y_{{\mathrm{{fixed}}}}$"
        )
        ax.set_ylabel("Probability density")

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax
    
    def perturbation_response(
            self,
            variable,
            perturbations=np.arange(-0.5,0.51,0.1),
            base_values=None,
            sample_size=10000,
            relative=True
    ):
        """
        Calculate formula response under variable perturbation.

        Parameters
        ----------
        variable : str
            variable name

        perturbations : array
            relative perturbation amplitude

            e.g.
            [-0.2,-0.1,0,0.1,0.2]

        base_values : dict
            background values of other variables

        relative : bool
            return relative change of output

        Returns
        -------
        perturbation
        response
        """

        if base_values is None:
            base_values = {}

        var_index = self.variable_names.index(variable)


        # -----------------------------
        # generate base samples
        # -----------------------------
        X = np.zeros(
            (sample_size,
             len(self.variable_names))
        )


        for i,name in enumerate(
            self.variable_names
        ):

            if name in base_values:

                X[:,i] = base_values[name]

            else:

                low,high=self.bounds[i]

                X[:,i] = np.random.uniform(
                    low,
                    high,
                    sample_size
                )


        # base output

        Y0 = self.model(X)


        response=[]


        # -----------------------------
        # perturb variable
        # -----------------------------

        x0 = X[:,var_index].copy()


        for p in perturbations:

            Xp=X.copy()


            Xp[:,var_index] = (
                x0*(1+p)
            )


            Yp=self.model(Xp)


            if relative:

                delta = (
                    np.mean(Yp-Y0)
                    /
                    np.mean(np.abs(Y0))
                )

            else:

                delta=np.mean(Yp-Y0)


            response.append(delta)



        return (
            np.asarray(perturbations),
            np.asarray(response)
        )
    
    def plot_perturbation_response(
            self,
            variable,
        save_path=None,
            perturbations=None,
            sample_size=10000,
            ax=None
    ):


        if perturbations is None:

            perturbations=np.array(
                [
                    -0.5,
                    -0.4,
                    -0.3,
                    -0.2,
                    -0.1,
                    0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5
                ]
            )


        x,y=self.perturbation_response(
            variable,
            perturbations,
            sample_size=sample_size
        )


        if ax is None:

            fig,ax=plt.subplots(
                figsize=(6,4)
            )


        ax.plot(
            x*100,
            y*100,
            marker="o",
            linewidth=2
        )


        ax.axhline(
            0,
            linestyle="--"
        )

        ax.axvline(
            0,
            linestyle="--"
        )


        ax.set_xlabel(
            f"$\Delta {variable}$ (%)"
        )


        ax.set_ylabel(
            r"$\Delta Y/Y_0$ (%)"
        )


        ax.set_title(
            f"Perturbation response of {variable}"
        )

        self._format_axis(ax)
        fig.tight_layout()

        self._save_figure(fig, save_path, dpi)

        if show:
            plt.show()

        return fig, ax

    # ========================================================
    # Run all steps with one call
    # ========================================================
    def run_all(
        self,
        ablation_sample_size: int = 100000,
        fixed_values: Optional[dict[str, float]] = None,
        make_plots: bool = True,
        output_directory: Optional[Union[str, Path]] = None,
        show_plots: bool = True,
    ) -> dict:
        """
        Run Sobol analysis, ablation analysis, result saving, and the main plots with one call.
        """

        print(
            "Expected Sobol model evaluations: "
            f"{self.expected_model_evaluations:,}"
        )

        sensitivity = self.run_sobol()

        ablation = self.run_ablation_all(
            fixed_values=fixed_values,
            sample_size=ablation_sample_size,
        )

        output_dir = None

        if output_directory is not None:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.save_results(output_dir)

        if make_plots:
            def path(name: str) -> Optional[Path]:
                if output_dir is None:
                    return None
                return output_dir / name

            self.plot_sobol_indices(
                save_path=path("sobol_indices.png"),
                show=show_plots,
            )

            self.plot_variable_contributions(
                contribution="ST",
                save_path=path("total_contributions.png"),
                show=show_plots,
            )

            self.plot_contribution_decomposition(
                save_path=path("contribution_decomposition.png"),
                show=show_plots,
            )

            self.plot_weak_contributions(
                save_path=path("weak_contributions_log.png"),
                show=show_plots,
            )

            self.plot_ablation_contributions(
                save_path=path("ablation_contributions.png"),
                show=show_plots,
            )

            if self.calc_second_order:
                self.plot_interaction_matrix(
                    save_path=path("interaction_matrix.png"),
                    show=show_plots,
                )

        return {
            "sensitivity_table": sensitivity,
            "ablation_table": ablation,
            "second_order_table": self.second_order_table,
            "summary": self.summary(),
        }

    # ========================================================
    # Automatic summary
    # ========================================================
    def summary(
        self,
        weak_threshold: float = 0.01,
    ) -> str:

        self._require_sobol_results()

        table = self.sensitivity_table

        strongest = table.iloc[0]
        weak = table[table["ST"] < weak_threshold]
        interaction_driven = table[
            (table["S1_lower"] <= 0)
            & (table["ST_lower"] > 0)
        ]

        lines = [
            "Variance-contribution analysis summary",
            "=" * 42,
            (
                f"Number of variables: "
                f"{len(self.variable_names)}"
            ),
            (
                f"Sobol base sample size: "
                f"{self.base_sample_size}"
            ),
            (
                "Number of model evaluations: "
                f"{self.expected_model_evaluations}"
            ),
            (
                f"Model-output variance: "
                f"{np.var(self.Y_sobol, ddof=1):.6e}"
            ),
            "",
            (
                "Strongest variable: "
                f"{strongest['Variable']}"
            ),
            f"Strongest S1: {strongest['S1']:.6e}",
            f"Strongest ST: {strongest['ST']:.6e}",
            "",
        ]

        if len(weak) > 0:
            lines.append(
                f"Weak variables with ST < {weak_threshold:g}:"
            )
            for _, row in weak.iterrows():
                lines.append(
                    f"  - {row['Variable']}: "
                    f"S1={row['S1']:.3e}, "
                    f"ST={row['ST']:.3e}"
                )
        else:
            lines.append(
                f"No variable has ST < {weak_threshold:g}."
            )

        lines.append("")

        if len(interaction_driven) > 0:
            lines.append(
                "Variables whose effects are mainly interaction-driven:"
            )
            for _, row in interaction_driven.iterrows():
                lines.append(
                    f"  - {row['Variable']}: "
                    f"ST-S1={row['Interaction']:.3e}"
                )
        else:
            lines.append(
                "No clearly resolved interaction-dominated variable."
            )

        return "\n".join(lines)

    # ========================================================
    # Save the results
    # ========================================================
    def save_results(
        self,
        output_directory: Union[str, Path],
    ) -> None:
        """Save all computed tables and text reports."""

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        if self.sensitivity_table is not None:
            self.sensitivity_table.to_csv(
                output_directory / "sobol_sensitivity.csv",
                index=False,
            )

        if self.second_order_table is not None:
            self.second_order_table.to_csv(
                output_directory / "sobol_second_order.csv"
            )

        if self.second_order_conf_table is not None:
            self.second_order_conf_table.to_csv(
                output_directory
                / "sobol_second_order_confidence.csv"
            )

        if self.ablation_table is not None:
            self.ablation_table.to_csv(
                output_directory / "ablation_results.csv",
                index=False,
            )

        if self.sensitivity_table is not None:
            report = self.summary()

            with open(
                output_directory / "analysis_summary.txt",
                "w",
                encoding="utf-8",
            ) as file:
                file.write(report)

    @staticmethod
    def _save_figure(
        fig: plt.Figure,
        save_path: Optional[Union[str, Path]],
        dpi: int,
    ) -> None:

        if save_path is None:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        pass
    
    pass

##
def physical_formula(X):
    """
    X[:, 0] = X1
    X[:, 1] = X2
    X[:, 2] = X3
    X[:, 3] = gamma
    """
    bz = X[:, 0]
    r = X[:, 1]
    sini = X[:, 2]
    cosi = X[:, 3]
    sind = X[:, 4]
    cosd = X[:, 5]
    dudz = X[:, 6]
    dvdz = X[:, 7]
    
    
    f = (cosd * cosi) / (r + sind*sini + (1.0053264/r))
    g = (sind + (sini*cosd) / r) * (-1.0127895) * cosi * (1 / (r+0.23963471))
    pseudo_dwidz = f*dudz + g*dvdz
    
    
    h = (0.16050342 - (bz / -0.0039426372));

    # g = (dwidz / dwidz) - ((((dwidz * -0.06478512) / ((dwidz * (dwidz - 0.15226728)) - -0.117391296)) - -0.098632485) * dwidz)
    i = 1 - 0.098632485 * pseudo_dwidz 
    # + 0.06478512 * pseudo_dwidz**2
    
    y = h * i

    return y




if __name__ == "__main__":
    
    # Construct the following based on the physical_formula function definition
    analyzer = VarianceContributionAnalyzer(
        model=physical_formula,

        variable_names=[
            "bz",
            "r",
            "sini",
            "cosi",
            "sind",
            "cosd",
            "dudz",
            "dvdz"
        ],

        bounds=[
            [-0.0001, 0.00015], #bz
            [4., 122.], #r
            [-0.94, 0.93], #sini
            [0.33, 0.99], #cosi
            [-0.57, 0.38], #sind
            [0.81, 0.99], #cosd
            [-77.82, 95.0], #dudz
            [-99.13, 103.41], #dvdz
        ],

        units=[
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],

        base_sample_size=4096,
        calc_second_order=True,
        num_resamples=2000,
        conf_level=0.95,
        random_seed=42,
        output_name=r"$Y$",
    )


    



    #######################
    # Perturbation response curves for different variables
#     variables=[
#         "bz",
#             "r",
#             # "sini",
#             # "cosi",
#             # "sind",
#             # "cosd",
#             "dudz",
#             "dvdz"
#     ]

#     fig,ax=plt.subplots(
#         figsize=(7,5)
#     )
#     data_dict = {}
#     for var in variables:

#         x,y=analyzer.perturbation_response(
#             var,
#             perturbations=np.linspace(
#                 -0.9,
#                 1.,
#                 50
#             ),
#             sample_size=50000
#         )
        

#         ax.plot(
#             x*100,
#             y*100,
#             marker="",
#             label=var
#         )
#         data_dict[var] = [x, y]
#         pass
    
#     save_file_name = "./salib_analysis/perturbation.joblib"
#     joblib.dump(data_dict, save_file_name)
    
#     loaded_data = joblib.load(save_file_name)
#     breakpoint()
    
#     ax.set_xlabel(
#         "Variable perturbation (%)"
#     )
#     ax.set_ylabel(
#         "Relative response of formula (%)"
#     )
#     ax.set_ylim(-10, 10)
#     ax.legend()
#     fig.savefig(
#             "./salib_analysis/variable_perturbation.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
    
    
    ###############
    # Variable contribution curves
    # analyzer.run_sobol()
    # analyzer.plot_contribution_decomposition(save_path="./salib_analysis/contribution_decomposition.png")

#     analyzer.run_all()
#     # Run Sobol analysis
#     sensitivity_table = analyzer.run_sobol()
#     print(sensitivity_table)

#     # analyzer.run_all()

#     analyzer.plot_sobol_indices(
#         sort_by="ST",
#         figsize=(8,5),
#         save_path="./sobol_indices.png"
#     )
    
#     analyzer.plot_variable_contributions(
#     contribution="ST",
#     percentage=True,
#         save_path="./variable_contributions.png"
# )


#
    results = analyzer.run_all(
        ablation_sample_size=100000,
    #     fixed_values={
    #         "bz": 0.,

    #     },
        fixed_values=None,
        make_plots=True,
        output_directory="./salib_analysis",
        show_plots=True,
    )
    # breakpoint()
    save_file_name = "./salib_analysis/ablation_analyzer.joblib"
    
    # After execution, the object has attributes such as sensitivity table and ablation details
    # Save its attributes as a dictionary 
    
    data_dict = {}
    data_dict["sensitivity_table"] = analyzer.sensitivity_table
    data_dict["ablation_table"] = analyzer.ablation_table
    data_dict["ablation_details"] = analyzer.ablation_details
    
    joblib.dump(data_dict, save_file_name)

    # loaded_data = joblib.load(save_file_name)
    # breakpoint()


    # Inspect the corresponding variable contents below
    #
    # print(analyzer.interpret_variable("bz"))


    # analyzer.plot_variable_interactions(
    #     "cosi",save_path="./cosi_ablation_result/variable_interactions.png"
    # )

    # analyzer.plot_ablation_comparison(
    #     "cosi",save_path="./cosi_ablation_result/ablation_comparison.png"
    # )

    # analyzer.plot_ablation_delta(
    #     "cosi",
    #     log_y=True,save_path="./cosi_ablation_result/ablation_delta.png"
    # )