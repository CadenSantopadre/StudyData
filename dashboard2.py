import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy import stats
from scipy.linalg import eigvals

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QStackedWidget,
    QMessageBox,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# DATA LOADING
# ============================================================

CSV_FILE = "study.csv"

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    raise SystemExit(
        f"Could not find '{CSV_FILE}'. Place it in the same folder as this script."
    )

REQUIRED_COLUMNS = [
    "Hours",
    "Velocity",
    "Happiness",
    "Stress",
]

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise SystemExit(
        "The following required columns are missing from study.csv:\n"
        + ", ".join(missing_columns)
    )

df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
df = df.dropna().reset_index(drop=True)

if len(df) < 5:
    raise SystemExit("study.csv must contain at least five complete data rows.")


# ============================================================
# GENERAL ANALYSIS FUNCTIONS
# ============================================================

def align_lagged_data(dataframe, x_name, y_name, lag):
    """
    Aligns x and y using a lag.

    lag = 0:
        x[t] compared with y[t]

    lag > 0:
        x[t] compared with y[t + lag]

    lag < 0:
        x[t - lag] compared with y[t]
    """

    x = dataframe[x_name].to_numpy(dtype=float)
    y = dataframe[y_name].to_numpy(dtype=float)

    if lag > 0:
        x_aligned = x[:-lag]
        y_aligned = y[lag:]

    elif lag < 0:
        x_aligned = x[-lag:]
        y_aligned = y[:lag]

    else:
        x_aligned = x
        y_aligned = y

    valid = np.isfinite(x_aligned) & np.isfinite(y_aligned)

    return (
        x_aligned[valid],
        y_aligned[valid],
    )


def calculate_statistics(x, y):
    """
    Returns common statistics for two aligned numerical arrays.
    """

    if len(x) < 2 or len(y) < 2:
        return {
            "N": len(x),
            "Pearson r": np.nan,
            "Pearson p": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
            "R²": np.nan,
        }

    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "N": len(x),
            "Pearson r": np.nan,
            "Pearson p": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
            "R²": np.nan,
        }

    correlation = stats.pearsonr(x, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "N": len(x),
        "Pearson r": correlation.statistic,
        "Pearson p": correlation.pvalue,
        "Slope": slope,
        "Intercept": intercept,
        "R²": r_value ** 2,
    }


def create_state_matrices(dataframe, variables):
    """
    Creates a one-step state-transition dataset:

        x[t+1] = A x[t] + b

    X contains the current state.
    Y contains the next state.
    """

    current_state = dataframe[variables].iloc[:-1].to_numpy(dtype=float)
    next_state = dataframe[variables].iloc[1:].to_numpy(dtype=float)

    return current_state, next_state


def fit_linear_dynamics(dataframe, variables):
    """
    Fits:

        x[t+1] = A x[t] + b

    Returns:
        A: transition matrix
        b: intercept vector
        predictions: predicted next states
        actual: actual next states
        model: sklearn model
    """

    X, Y = create_state_matrices(dataframe, variables)

    model = LinearRegression()
    model.fit(X, Y)

    predictions = model.predict(X)

    A = model.coef_
    b = model.intercept_

    return {
        "A": A,
        "b": b,
        "predictions": predictions,
        "actual": Y,
        "model": model,
        "X": X,
        "Y": Y,
    }


def fit_polynomial_dynamics(
    dataframe,
    variables,
    degree=2,
):
    """
    Fits:

        x[t+1] = f(x[t])

    using polynomial features.

    Returns the fitted model, predictions, feature names,
    and coefficient matrix.
    """

    X, Y = create_state_matrices(
        dataframe,
        variables,
    )

    polynomial_features = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_polynomial = polynomial_features.fit_transform(X)

    regression_model = LinearRegression()
    regression_model.fit(
        X_polynomial,
        Y,
    )

    predictions = regression_model.predict(
        X_polynomial
    )

    feature_names = (
        polynomial_features.get_feature_names_out(
            variables
        )
    )

    coefficient_matrix = regression_model.coef_

    return {
        "model": Pipeline(
            [
                (
                    "polynomial_features",
                    polynomial_features,
                ),
                (
                    "regression",
                    regression_model,
                ),
            ]
        ),
        "polynomial_features": polynomial_features,
        "regression_model": regression_model,
        "predictions": predictions,
        "actual": Y,
        "X": X,
        "Y": Y,
        "feature_names": feature_names,
        "coefficient_matrix": coefficient_matrix,
    }

def calculate_prediction_metrics(actual, predicted):
    """
    Calculates metrics across all output variables.
    """

    actual_flat = actual.reshape(-1)
    predicted_flat = predicted.reshape(-1)

    return {
        "RMSE": np.sqrt(
            mean_squared_error(actual_flat, predicted_flat)
        ),
        "MAE": mean_absolute_error(
            actual_flat,
            predicted_flat,
        ),
        "R²": r2_score(
            actual_flat,
            predicted_flat,
        ),
    }


def chronological_split(dataframe, train_fraction=0.8):
    """
    Splits time-series data chronologically.

    No shuffling is performed.
    """

    split_index = int(len(dataframe) * train_fraction)

    split_index = max(2, split_index)
    split_index = min(len(dataframe) - 2, split_index)

    train = dataframe.iloc[:split_index].copy()
    test = dataframe.iloc[split_index:].copy()

    return train, test


# ============================================================
# MATPLOTLIB CANVAS
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(
            figsize=(8, 6),
            tight_layout=True,
        )

        super().__init__(self.figure)
        self.setParent(parent)

    def draw_polynomial_coefficients(
    self,
    coefficient_matrix,
    feature_names,
    output_variables,
    output_variable,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        output_index = output_variables.index(
            output_variable
        )

        coefficients = coefficient_matrix[
            output_index
        ]

        order = np.argsort(
            np.abs(coefficients)
        )[::-1]

        sorted_features = [
            feature_names[index]
            for index in order
        ]

        sorted_coefficients = [
            coefficients[index]
            for index in order
        ]

        # Limit the display to the largest terms
        # so the graph remains readable.
        max_terms = min(
            20,
            len(sorted_features),
        )

        sorted_features = sorted_features[
            :max_terms
        ]

        sorted_coefficients = sorted_coefficients[
            :max_terms
        ]

        y_positions = np.arange(
            len(sorted_features)
        )

        ax.barh(
            y_positions,
            sorted_coefficients,
        )

        ax.set_yticks(
            y_positions
        )

        ax.set_yticklabels(
            sorted_features
        )

        ax.invert_yaxis()

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Polynomial feature")

        ax.set_title(
            f"Largest polynomial coefficients\n"
            f"Predicting next {output_variable}"
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

        self.draw()

    def clear_plot(self):
        self.figure.clear()

    def draw_scatter(
        self,
        x,
        y,
        x_name,
        y_name,
        lag,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        ax.scatter(
            x,
            y,
            alpha=0.75,
            edgecolors="none",
        )

        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept, *_ = stats.linregress(x, y)

            x_line = np.linspace(
                np.min(x),
                np.max(x),
                100,
            )

            y_line = slope * x_line + intercept

            ax.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=2,
                label="Linear fit",
            )

            ax.legend()

        ax.set_title(
            f"{y_name} vs {x_name} | Lag = {lag}"
        )

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(alpha=0.25)

        self.draw()

    def draw_time_series(
        self,
        dataframe,
        x_name,
        y_name,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        ax.plot(
            dataframe.index,
            dataframe[x_name],
            label=x_name,
            linewidth=2,
        )

        ax.plot(
            dataframe.index,
            dataframe[y_name],
            label=y_name,
            linewidth=2,
        )

        ax.set_title(
            f"Time series: {x_name} and {y_name}"
        )

        ax.set_xlabel("Time index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.25)

        self.draw()

    def draw_cross_correlation(
        self,
        dataframe,
        x_name,
        y_name,
        max_lag,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        lags = np.arange(
            -max_lag,
            max_lag + 1,
        )

        correlations = []

        for lag in lags:
            x, y = align_lagged_data(
                dataframe,
                x_name,
                y_name,
                int(lag),
            )

            if len(x) < 2:
                correlations.append(np.nan)
                continue

            if np.std(x) == 0 or np.std(y) == 0:
                correlations.append(np.nan)
                continue

            correlations.append(
                np.corrcoef(x, y)[0, 1]
            )

        ax.plot(
            lags,
            correlations,
            marker="o",
            linewidth=2,
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"Cross-correlation: {x_name} vs {y_name}"
        )

        ax.set_xlabel("Lag")
        ax.set_ylabel("Pearson correlation")
        ax.grid(alpha=0.25)

        self.draw()

    def draw_matrix(
        self,
        matrix,
        labels,
        title,
        annotate=True,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
        )

        self.figure.colorbar(
            image,
            ax=ax,
            label="Coefficient",
        )

        ax.set_xticks(
            np.arange(len(labels))
        )

        ax.set_yticks(
            np.arange(len(labels))
        )

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel("Input variable")
        ax.set_ylabel("Output variable")
        ax.set_title(title)

        if annotate:
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    ax.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

        self.draw()

    def draw_eigenvalues(
        self,
        eigenvalues,
        title,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111)

        theta = np.linspace(
            0,
            2 * np.pi,
            500,
        )

        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            linewidth=1.5,
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            s=100,
            zorder=5,
        )

        for index, value in enumerate(eigenvalues):
            ax.annotate(
                f"λ{index + 1}",
                (
                    np.real(value),
                    np.imag(value),
                ),
            )

        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)

        self.draw()

    def draw_actual_vs_predicted(
        self,
        actual,
        predicted,
        variable_names,
        title,
    ):
        self.clear_plot()

        number_of_variables = actual.shape[1]

        rows = int(np.ceil(number_of_variables / 2))
        columns = min(2, number_of_variables)

        axes = self.figure.subplots(
            rows,
            columns,
            squeeze=False,
        )

        axes = axes.flatten()

        for index, variable in enumerate(variable_names):
            ax = axes[index]

            ax.scatter(
                actual[:, index],
                predicted[:, index],
                alpha=0.75,
            )

            minimum = min(
                np.min(actual[:, index]),
                np.min(predicted[:, index]),
            )

            maximum = max(
                np.max(actual[:, index]),
                np.max(predicted[:, index]),
            )

            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--",
                linewidth=1.5,
            )

            ax.set_title(variable)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.25)

        for unused_axis in axes[number_of_variables:]:
            unused_axis.remove()

        self.figure.suptitle(title)

        self.draw()

    def draw_response_surface(
        self,
        model,
        dataframe,
        variables,
        input_x,
        input_y,
        output_variable,
        degree,
    ):
        self.clear_plot()

        ax = self.figure.add_subplot(111, projection="3d")

        x_values = dataframe[input_x].to_numpy()
        y_values = dataframe[input_y].to_numpy()

        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)

        x_grid = np.linspace(
            x_min,
            x_max,
            35,
        )

        y_grid = np.linspace(
            y_min,
            y_max,
            35,
        )

        grid_x, grid_y = np.meshgrid(
            x_grid,
            y_grid,
        )

        base_state = dataframe[variables].mean().to_numpy()

        prediction_inputs = []

        for x_value, y_value in zip(
            grid_x.reshape(-1),
            grid_y.reshape(-1),
        ):
            state = base_state.copy()

            state[variables.index(input_x)] = x_value
            state[variables.index(input_y)] = y_value

            prediction_inputs.append(state)

        prediction_inputs = np.asarray(
            prediction_inputs
        )

        predictions = model.predict(
            prediction_inputs
        )

        output_index = variables.index(
            output_variable
        )

        prediction_surface = predictions[
            :, output_index
        ].reshape(grid_x.shape)

        ax.plot_surface(
            grid_x,
            grid_y,
            prediction_surface,
            alpha=0.75,
            cmap="viridis",
        )

        ax.set_xlabel(input_x)
        ax.set_ylabel(input_y)
        ax.set_zlabel(
            f"Predicted next {output_variable}"
        )

        ax.set_title(
            f"Polynomial response surface\n"
            f"{output_variable} from {input_x} and {input_y}"
        )

        self.draw()


# ============================================================
# MAIN WINDOW
# ============================================================

class StudyDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Study Dynamics Dashboard"
        )

        self.resize(
            1400,
            900,
        )

        self.variables = REQUIRED_COLUMNS.copy()

        self.current_linear_result = None
        self.current_nonlinear_result = None
        self.current_validation_result = None

        self.setup_ui()
        self.update_relationship_analysis()

    # --------------------------------------------------------
    # UI SETUP
    # --------------------------------------------------------

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(
            central_widget
        )

        # Top-level mode selector
        mode_layout = QHBoxLayout()

        mode_label = QLabel("Analysis mode:")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Relationships",
                "Linear dynamics",
                "Nonlinear dynamics",
                "Model validation",
            ]
        )

        self.mode_combo.currentIndexChanged.connect(
            self.change_analysis_mode
        )

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(
            self.mode_combo
        )
        mode_layout.addStretch()

        main_layout.addLayout(mode_layout)

        # Mode-specific menus
        self.mode_stack = QStackedWidget()

        self.relationship_page = (
            self.create_relationship_page()
        )

        self.linear_page = (
            self.create_linear_page()
        )

        self.nonlinear_page = (
            self.create_nonlinear_page()
        )

        self.validation_page = (
            self.create_validation_page()
        )

        self.mode_stack.addWidget(
            self.relationship_page
        )

        self.mode_stack.addWidget(
            self.linear_page
        )

        self.mode_stack.addWidget(
            self.nonlinear_page
        )

        self.mode_stack.addWidget(
            self.validation_page
        )

        main_layout.addWidget(
            self.mode_stack
        )

        # Main plot/statistics splitter
        splitter = QSplitter(
            Qt.Orientation.Vertical
        )

        self.canvas = PlotCanvas()

        splitter.addWidget(
            self.canvas
        )

        self.statistics_table = QTableWidget()
        self.statistics_table.setColumnCount(2)
        self.statistics_table.setHorizontalHeaderLabels(
            [
                "Statistic",
                "Value",
            ]
        )

        self.statistics_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )

        self.statistics_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )

        splitter.addWidget(
            self.statistics_table
        )

        splitter.setSizes(
            [
                600,
                250,
            ]
        )

        main_layout.addWidget(
            splitter
        )

        self.status_label = QLabel(
            "Ready."
        )

        main_layout.addWidget(
            self.status_label
        )

    # --------------------------------------------------------
    # RELATIONSHIPS PAGE
    # --------------------------------------------------------

    def create_relationship_page(self):
        page = QWidget()
        layout = QGridLayout(page)

        self.relationship_x_combo = QComboBox()
        self.relationship_x_combo.addItems(
            self.variables
        )

        self.relationship_y_combo = QComboBox()
        self.relationship_y_combo.addItems(
            self.variables
        )

        self.relationship_lag_spin = QSpinBox()
        self.relationship_lag_spin.setRange(
            -1000,
            1000,
        )
        self.relationship_lag_spin.setValue(0)

        self.relationship_plot_combo = QComboBox()
        self.relationship_plot_combo.addItems(
            [
                "Scatter",
                "Time series",
                "Cross-correlation",
            ]
        )

        self.relationship_max_lag_spin = QSpinBox()
        self.relationship_max_lag_spin.setRange(
            1,
            1000,
        )
        self.relationship_max_lag_spin.setValue(10)

        update_button = QPushButton(
            "Update relationship analysis"
        )

        update_button.clicked.connect(
            self.update_relationship_analysis
        )

        layout.addWidget(
            QLabel("X variable"),
            0,
            0,
        )

        layout.addWidget(
            self.relationship_x_combo,
            0,
            1,
        )

        layout.addWidget(
            QLabel("Y variable"),
            0,
            2,
        )

        layout.addWidget(
            self.relationship_y_combo,
            0,
            3,
        )

        layout.addWidget(
            QLabel("Lag"),
            1,
            0,
        )

        layout.addWidget(
            self.relationship_lag_spin,
            1,
            1,
        )

        layout.addWidget(
            QLabel("Plot"),
            1,
            2,
        )

        layout.addWidget(
            self.relationship_plot_combo,
            1,
            3,
        )

        layout.addWidget(
            QLabel("Maximum cross-correlation lag"),
            2,
            0,
        )

        layout.addWidget(
            self.relationship_max_lag_spin,
            2,
            1,
        )

        layout.addWidget(
            update_button,
            2,
            3,
        )

        return page

    def update_relationship_analysis(self):
        x_name = (
            self.relationship_x_combo.currentText()
        )

        y_name = (
            self.relationship_y_combo.currentText()
        )

        lag = (
            self.relationship_lag_spin.value()
        )

        plot_type = (
            self.relationship_plot_combo.currentText()
        )

        max_lag = (
            self.relationship_max_lag_spin.value()
        )

        if plot_type == "Scatter":
            x, y = align_lagged_data(
                df,
                x_name,
                y_name,
                lag,
            )

            self.canvas.draw_scatter(
                x,
                y,
                x_name,
                y_name,
                lag,
            )

            statistics = calculate_statistics(
                x,
                y,
            )

            self.display_statistics(
                statistics
            )

            self.status_label.setText(
                f"Scatter plot generated using {len(x)} aligned observations."
            )

        elif plot_type == "Time series":
            self.canvas.draw_time_series(
                df,
                x_name,
                y_name,
            )

            statistics = calculate_statistics(
                df[x_name].to_numpy(),
                df[y_name].to_numpy(),
            )

            self.display_statistics(
                statistics
            )

            self.status_label.setText(
                "Time-series plot generated."
            )

        elif plot_type == "Cross-correlation":
            self.canvas.draw_cross_correlation(
                df,
                x_name,
                y_name,
                max_lag,
            )

            self.display_statistics(
                {
                    "Maximum lag searched": max_lag,
                    "Observations": len(df),
                }
            )

            self.status_label.setText(
                "Cross-correlation generated."
            )

    # --------------------------------------------------------
    # LINEAR DYNAMICS PAGE
    # --------------------------------------------------------

    def create_linear_page(self):
        page = QWidget()
        layout = QGridLayout(page)

        self.linear_plot_combo = QComboBox()

        self.linear_plot_combo.addItems(
            [
                "Transition matrix",
                "Eigenvalues",
                "Actual vs predicted",
            ]
        )

        self.linear_fit_button = QPushButton(
            "Fit linear dynamics model"
        )

        self.linear_fit_button.clicked.connect(
            self.update_linear_dynamics
        )

        self.linear_show_intercept_checkbox = QCheckBox(
            "Show intercept vector in statistics"
        )

        self.linear_show_intercept_checkbox.setChecked(
            True
        )

        layout.addWidget(
            QLabel("Linear dynamics plot"),
            0,
            0,
        )

        layout.addWidget(
            self.linear_plot_combo,
            0,
            1,
        )

        layout.addWidget(
            self.linear_show_intercept_checkbox,
            1,
            0,
            1,
            2,
        )

        layout.addWidget(
            self.linear_fit_button,
            1,
            3,
        )

        return page

    def update_linear_dynamics(self):
        result = fit_linear_dynamics(
            df,
            self.variables,
        )

        self.current_linear_result = result

        A = result["A"]
        b = result["b"]

        eigenvalues_of_A = eigvals(A)

        plot_type = (
            self.linear_plot_combo.currentText()
        )

        if plot_type == "Transition matrix":
            self.canvas.draw_matrix(
                A,
                self.variables,
                "Linear transition matrix A",
            )

        elif plot_type == "Eigenvalues":
            self.canvas.draw_eigenvalues(
                eigenvalues_of_A,
                "Eigenvalues of transition matrix A",
            )

        elif plot_type == "Actual vs predicted":
            self.canvas.draw_actual_vs_predicted(
                result["actual"],
                result["predictions"],
                self.variables,
                "Linear model: actual vs predicted",
            )

        metrics = calculate_prediction_metrics(
            result["actual"],
            result["predictions"],
        )

        statistics = {
            "Model": "x[t+1] = A x[t] + b",
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R²": metrics["R²"],
            "Maximum eigenvalue magnitude": np.max(
                np.abs(eigenvalues_of_A)
            ),
        }

        for index, eigenvalue in enumerate(
            eigenvalues_of_A
        ):
            statistics[
                f"Eigenvalue {index + 1}"
            ] = eigenvalue

        if self.linear_show_intercept_checkbox.isChecked():
            for index, value in enumerate(b):
                statistics[
                    f"Intercept: {self.variables[index]}"
                ] = value

        spectral_radius = np.max(
            np.abs(eigenvalues_of_A)
        )

        if spectral_radius < 1:
            stability = "Locally stable in the fitted discrete model."

        elif np.isclose(spectral_radius, 1):
            stability = (
                "Marginal or borderline stability. "
                "Interpret cautiously."
            )

        else:
            stability = (
                "Unstable or expanding behavior in the fitted model."
            )

        statistics["Stability"] = stability

        self.display_statistics(
            statistics
        )

        self.status_label.setText(
            "Linear dynamics model fitted successfully."
        )

    # --------------------------------------------------------
    # NONLINEAR DYNAMICS PAGE
    # --------------------------------------------------------

    def create_nonlinear_page(self):
        page = QWidget()
        layout = QGridLayout(page)

        self.nonlinear_degree_spin = QSpinBox()
        self.nonlinear_degree_spin.setRange(
            2,
            5,
        )
        self.nonlinear_degree_spin.setValue(2)

        self.nonlinear_plot_combo = QComboBox()
        self.nonlinear_plot_combo.addItems(
            [
                "Actual vs predicted",
                "Response surface",
                "Polynomial coefficients",
            ]
        )

        self.nonlinear_x_combo = QComboBox()
        self.nonlinear_x_combo.addItems(
            self.variables
        )

        self.nonlinear_y_combo = QComboBox()
        self.nonlinear_y_combo.addItems(
            self.variables
        )

        self.nonlinear_output_combo = QComboBox()
        self.nonlinear_output_combo.addItems(
            self.variables
        )

        self.nonlinear_fit_button = QPushButton(
            "Fit polynomial dynamics model"
        )

        self.nonlinear_fit_button.clicked.connect(
            self.update_nonlinear_dynamics
        )

        layout.addWidget(
            QLabel("Polynomial degree"),
            0,
            0,
        )

        layout.addWidget(
            self.nonlinear_degree_spin,
            0,
            1,
        )

        layout.addWidget(
            QLabel("Plot"),
            0,
            2,
        )

        layout.addWidget(
            self.nonlinear_plot_combo,
            0,
            3,
        )

        layout.addWidget(
            QLabel("Surface X"),
            1,
            0,
        )

        layout.addWidget(
            self.nonlinear_x_combo,
            1,
            1,
        )

        layout.addWidget(
            QLabel("Surface Y"),
            1,
            2,
        )

        layout.addWidget(
            self.nonlinear_y_combo,
            1,
            3,
        )

        layout.addWidget(
            QLabel("Surface output"),
            2,
            0,
        )

        layout.addWidget(
            self.nonlinear_output_combo,
            2,
            1,
        )

        layout.addWidget(
            self.nonlinear_fit_button,
            2,
            3,
        )

        return page

    def update_nonlinear_dynamics(self):
        degree = (
            self.nonlinear_degree_spin.value()
        )

        plot_type = (
            self.nonlinear_plot_combo.currentText()
        )

        result = fit_polynomial_dynamics(
            df,
            self.variables,
            degree=degree,
        )

        self.current_nonlinear_result = result

        metrics = calculate_prediction_metrics(
            result["actual"],
            result["predictions"],
        )

        if plot_type == "Actual vs predicted":
            self.canvas.draw_actual_vs_predicted(
                result["actual"],
                result["predictions"],
                self.variables,
                f"Polynomial degree {degree}: actual vs predicted",
            )

        elif plot_type == "Response surface":
            input_x = (
                self.nonlinear_x_combo.currentText()
            )

            input_y = (
                self.nonlinear_y_combo.currentText()
            )

            output_variable = (
                self.nonlinear_output_combo.currentText()
            )

            if input_x == input_y:
                QMessageBox.warning(
                    self,
                    "Invalid response surface",
                    "Surface X and Surface Y must be different variables.",
                )
                return

            self.canvas.draw_response_surface(
                result["model"],
                df,
                self.variables,
                input_x,
                input_y,
                output_variable,
                degree,
            )
        elif plot_type == "Polynomial coefficients":
            output_variable = (
                self.nonlinear_output_combo.currentText()
            )

            self.canvas.draw_polynomial_coefficients(
                result["coefficient_matrix"],
                result["feature_names"],
                self.variables,
                output_variable,
            )

        statistics = {
            "Model": (
                f"Polynomial state model, degree {degree}"
            ),
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R²": metrics["R²"],
            "Training observations": len(result["X"]),
        }

        self.display_statistics(
            statistics
        )

        self.status_label.setText(
            "Polynomial nonlinear dynamics model fitted."
        )

    # --------------------------------------------------------
    # MODEL VALIDATION PAGE
    # --------------------------------------------------------

    def create_validation_page(self):
        page = QWidget()
        layout = QGridLayout(page)

        self.validation_train_fraction_spin = QDoubleSpinBox()
        self.validation_train_fraction_spin.setRange(
            0.5,
            0.95,
        )
        self.validation_train_fraction_spin.setSingleStep(
            0.05
        )
        self.validation_train_fraction_spin.setValue(
            0.8
        )

        self.validation_degree_spin = QSpinBox()
        self.validation_degree_spin.setRange(
            2,
            5,
        )
        self.validation_degree_spin.setValue(
            2
        )

        self.validation_model_combo = QComboBox()
        self.validation_model_combo.addItems(
            [
                "Linear",
                "Polynomial",
            ]
        )

        self.validation_plot_combo = QComboBox()
        self.validation_plot_combo.addItems(
            [
                "Actual vs predicted",
                "Metrics only",
            ]
        )

        self.validation_run_button = QPushButton(
            "Run chronological validation"
        )

        self.validation_run_button.clicked.connect(
            self.update_validation
        )

        layout.addWidget(
            QLabel("Training fraction"),
            0,
            0,
        )

        layout.addWidget(
            self.validation_train_fraction_spin,
            0,
            1,
        )

        layout.addWidget(
            QLabel("Polynomial degree"),
            0,
            2,
        )

        layout.addWidget(
            self.validation_degree_spin,
            0,
            3,
        )

        layout.addWidget(
            QLabel("Model"),
            1,
            0,
        )

        layout.addWidget(
            self.validation_model_combo,
            1,
            1,
        )

        layout.addWidget(
            QLabel("Plot"),
            1,
            2,
        )

        layout.addWidget(
            self.validation_plot_combo,
            1,
            3,
        )

        layout.addWidget(
            self.validation_run_button,
            2,
            3,
        )

        return page

    def update_validation(self):
        train_fraction = (
            self.validation_train_fraction_spin.value()
        )

        degree = (
            self.validation_degree_spin.value()
        )

        selected_model = (
            self.validation_model_combo.currentText()
        )

        plot_type = (
            self.validation_plot_combo.currentText()
        )

        train_df, test_df = chronological_split(
            df,
            train_fraction=train_fraction,
        )

        train_X, train_Y = create_state_matrices(
            train_df,
            self.variables,
        )

        test_X, test_Y = create_state_matrices(
            test_df,
            self.variables,
        )

        models_to_evaluate = {}

        if selected_model in [
            "Linear",
            "Compare linear and polynomial",
        ]:
            linear_model = LinearRegression()
            linear_model.fit(
                train_X,
                train_Y,
            )

            linear_predictions = (
                linear_model.predict(test_X)
            )

            models_to_evaluate["Linear"] = {
                "model": linear_model,
                "predictions": linear_predictions,
            }

        if selected_model in [
            "Polynomial",
            "Compare linear and polynomial",
        ]:
            polynomial_model = Pipeline(
                [
                    (
                        "polynomial_features",
                        PolynomialFeatures(
                            degree=degree,
                            include_bias=False,
                        ),
                    ),
                    (
                        "regression",
                        LinearRegression(),
                    ),
                ]
            )

            polynomial_model.fit(
                train_X,
                train_Y,
            )

            polynomial_predictions = (
                polynomial_model.predict(test_X)
            )

            models_to_evaluate["Polynomial"] = {
                "model": polynomial_model,
                "predictions": polynomial_predictions,
            }

        statistics = {
            "Training observations": len(train_X),
            "Testing observations": len(test_X),
            "Training fraction": train_fraction,
        }

        for model_name, result in models_to_evaluate.items():
            metrics = calculate_prediction_metrics(
                test_Y,
                result["predictions"],
            )

            statistics[
                f"{model_name} RMSE"
            ] = metrics["RMSE"]

            statistics[
                f"{model_name} MAE"
            ] = metrics["MAE"]

            statistics[
                f"{model_name} R²"
            ] = metrics["R²"]

        if plot_type == "Actual vs predicted":
            # If the user chose to compare, you might want to plot the Polynomial model 
            # (which changes with degrees) or loop through and plot both.
            if selected_model == "Compare linear and polynomial":
                # Option A: Default to displaying the dynamic Polynomial model on the graph
                model_to_plot = "Polynomial"
            else:
                model_to_plot = selected_model

            if model_to_plot in models_to_evaluate:
                chosen_model = models_to_evaluate[model_to_plot]
                
                self.canvas.draw_actual_vs_predicted(
                    test_Y,
                    chosen_model["predictions"],
                    self.variables,
                    f"Validation: {model_to_plot}",
                )


        self.current_validation_result = {
            "train": train_df,
            "test": test_df,
            "models": models_to_evaluate,
            "actual": test_Y,
        }

        self.display_statistics(
            statistics
        )

        self.status_label.setText(
            "Chronological model validation completed."
        )

    # --------------------------------------------------------
    # MODE SWITCHING
    # --------------------------------------------------------

    def change_analysis_mode(self, index):
        self.mode_stack.setCurrentIndex(
            index
        )

        if index == 0:
            self.update_relationship_analysis()

        elif index == 1:
            self.update_linear_dynamics()

        elif index == 2:
            self.update_nonlinear_dynamics()

        elif index == 3:
            self.update_validation()

    # --------------------------------------------------------
    # STATISTICS TABLE
    # --------------------------------------------------------

    def display_statistics(self, statistics):
        self.statistics_table.clearContents()
        self.statistics_table.setRowCount(
            len(statistics)
        )

        for row, (name, value) in enumerate(
            statistics.items()
        ):
            self.statistics_table.setItem(
                row,
                0,
                QTableWidgetItem(
                    str(name)
                ),
            )

            if isinstance(value, complex):
                formatted_value = (
                    f"{value.real:.6f} "
                    f"+ {value.imag:.6f}i"
                )

            elif isinstance(value, float):
                if np.isnan(value):
                    formatted_value = "NaN"
                else:
                    formatted_value = (
                        f"{value:.6f}"
                    )

            else:
                formatted_value = str(value)

            self.statistics_table.setItem(
                row,
                1,
                QTableWidgetItem(
                    formatted_value
                ),
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)

    window = StudyDashboard()
    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()