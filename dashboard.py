
import sys
import numpy as np
import pandas as pd

from scipy import stats

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QGridLayout, QLabel, 
    QComboBox, QSpinBox, QGroupBox, 
    QSplitter, QTableWidget, QTableWidgetItem, 
    QHeaderView,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure

df = pd.read_csv("study.csv")
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

def align_lagged_data(dataframe, x_name, y_name, lag):
    x = dataframe[x_name].to_numpy(dtype=float)
    y = dataframe[y_name].to_numpy(dtype=float)

    if lag > 0: # If lag is positive, shift y backward relative to x by removing the last lag elements of x and the first lag elements of y.
        x_aligned = x[:-lag]
        y_aligned = y[lag:]

    elif lag < 0: #If lag is negative, shift x backward relative to y by remoivng the last lag elements of y and the first lag elements of x
        x_aligned = x[-lag:]
        y_aligned = y[:lag]

    else: #If no lag, then let them be normal
        x_aligned = x
        y_aligned = y

    return x_aligned, y_aligned

def calculate_statistics(x, y):
    """
    Calculate all statistics for aligned X and Y.
    """

    n = len(x)

    result = {
        "N": n,
        "Pearson r": np.nan,
        "Pearson p": np.nan,
        "Slope": np.nan,
        "Intercept": np.nan,
        "R²": np.nan,
    }

    if n < 3: #Don't mess with stupid sample sizes
        return result

    if np.std(x) == 0 or np.std(y) == 0: #Or things that are "perfect" anyways
        return result

    pearson = stats.pearsonr(x, y)

    result["Pearson r"] = pearson.statistic
    result["Pearson p"] = pearson.pvalue

    regression = stats.linregress(x, y)

    result["Slope"] = regression.slope
    result["Intercept"] = regression.intercept
    result["R²"] = regression.rvalue ** 2

    return result

#Now we're dealing with matplotlib
class PlotCanvas(FigureCanvas):

    def __init__(self):
        self.figure = Figure(figsize=(8, 5))
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)

    def clear_plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def draw_scatter(self, x, y, x_name, y_name, lag):
        self.clear_plot()

        self.ax.scatter(
            x, y,
            alpha=0.80,
        )

        if len(x) >= 3 and np.std(x) > 0:
            regression = stats.linregress(x, y)

            x_line = np.linspace(
                np.min(x), np.max(x), 100
            )

            y_line = (
                regression.intercept
                + regression.slope * x_line
            )

            self.ax.plot(
                x_line,
                y_line,
                linestyle="--",
            )

        self.ax.set_xlabel(x_name)
        self.ax.set_ylabel(f"{y_name} (lag {lag})")
        self.ax.set_title(
            f"{x_name} vs {y_name} | Lag = {lag}"
        )

        self.ax.grid()
        self.figure.tight_layout()
        self.draw()

    def draw_time_series(
        self, dataframe, x_name, y_name, lag
    ):
        self.clear_plot()

        x = dataframe[x_name].to_numpy(dtype=float)
        y = dataframe[y_name].to_numpy(dtype=float)

        if lag > 0:
            x = x[:-lag]
            y = y[lag:]
        elif lag < 0:
            x = x[-lag:]
            y = y[:lag]

        self.ax.plot(
            x,
            label=f"{x_name}",
        )

        self.ax.plot(
            y,
            label=f"{y_name} (lag {lag})",
        )

        self.ax.set_xlabel("Time (Days)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Lagged time-series comparison")
        self.ax.legend()

        self.figure.tight_layout()
        self.draw()

    def draw_cross_correlation(
        self, dataframe, x_name, y_name, max_lag
    ):
        self.clear_plot()

        lags = range(-max_lag, max_lag + 1)
        correlations = []

        for lag in lags:
            x_aligned, y_aligned = align_lagged_data(
                dataframe, x_name, y_name, lag
            )

            if len(x_aligned) >= 3:
                r = stats.pearsonr(
                    x_aligned, y_aligned
                ).statistic
            else:
                r = np.nan

            correlations.append(r)

        self.ax.axhline(
            0, linestyle="--", linewidth=1
        )

        self.ax.plot(
            list(lags),
            correlations,
            marker="o",
            linewidth=1.8
        )

        self.ax.set_xlabel("Lag")
        self.ax.set_ylabel("Pearson correlation")
        self.ax.set_title(
            f"Cross-correlation: {x_name} vs {y_name}"
        )

        self.ax.grid(alpha=0.25)

        self.figure.tight_layout()
        self.draw()


#Now we're dealing with the actual window that pops up
class StudyDashboard(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Study Dynamics")
        self.resize(1250, 800)

        self.setup_ui()
        self.update_analysis()

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        controls = QGroupBox("Analysis controls")
        controls_layout = QGridLayout(controls)

        # X variable
        controls_layout.addWidget(
            QLabel("Variable X:"), 0, 0 #The extra numbers here specify what row, col it goes in
        )

        self.x_combo = QComboBox()
        self.x_combo.addItems(numeric_columns)
        self.x_combo.currentTextChanged.connect(
            self.update_analysis
        )

        controls_layout.addWidget(
            self.x_combo, 0, 1
        )

        # Y variable
        controls_layout.addWidget(
            QLabel("Variable Y:"), 0, 2
        )

        self.y_combo = QComboBox()
        self.y_combo.addItems(numeric_columns)

        self.y_combo.currentTextChanged.connect(
            self.update_analysis
        )

        controls_layout.addWidget(
            self.y_combo, 0, 3
        )

        controls_layout.addWidget(
            QLabel("Lag:"), 1, 0
        )

        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(-100, 100)
        self.lag_spin.setValue(0)
        self.lag_spin.valueChanged.connect(
            self.update_analysis
        )

        controls_layout.addWidget(
            self.lag_spin, 1, 1
        )

        controls_layout.addWidget(
            QLabel("Plot:"), 1, 2
        )

        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Scatter",
            "Time series",
            "Cross-correlation"
        ])

        self.plot_combo.currentTextChanged.connect(
            self.update_analysis
        )

        controls_layout.addWidget(
            self.plot_combo, 1, 3
        )

        # Maximum lag for cross-correlation
        controls_layout.addWidget(
            QLabel("Max cross-correlation lag:"), 2, 0
        )

        self.max_lag_spin = QSpinBox()
        self.max_lag_spin.setRange(1, 10000)
        self.max_lag_spin.setValue(10)
        self.max_lag_spin.valueChanged.connect(
            self.update_analysis
        )

        controls_layout.addWidget(
            self.max_lag_spin, 2, 1
        )

        main_layout.addWidget(controls)

        #We need a splitter to divide the plot from everything else
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.canvas = PlotCanvas()
        splitter.addWidget(self.canvas)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels([
            "Statistic", "Value"
        ])

        self.stats_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )

        self.stats_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        splitter.addWidget(self.stats_table)

        splitter.setSizes([850, 350])

        main_layout.addWidget(splitter)

        self.status_label = QLabel()
        main_layout.addWidget(self.status_label)

    def update_analysis(self):

        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        lag = self.lag_spin.value()
        plot_type = self.plot_combo.currentText()

        if x_name == y_name:
            self.status_label.setText(
                "Select two different variables for cross-variable analysis."
            )

        x, y = align_lagged_data(
            df, x_name, y_name, lag
        )

        results = calculate_statistics(x, y)

        self.stats_table.setRowCount(len(results))

        for row, (name, value) in enumerate(results.items()):

            self.stats_table.setItem(
                row, 0, QTableWidgetItem(name)
            )

            if isinstance(value, (float, np.floating)):
                if np.isfinite(value):
                    display_value = f"{value:.6f}"
                else:
                    display_value = "N/A"
            else:
                display_value = str(value)

            self.stats_table.setItem(
                row, 1, QTableWidgetItem(display_value)
            )

        if plot_type == "Scatter":
            self.canvas.draw_scatter(
                x, y, x_name, y_name, lag
            )

        elif plot_type == "Time series":
            self.canvas.draw_time_series(
                df, x_name, y_name, lag
            )

        elif plot_type == "Cross-correlation":
            self.canvas.draw_cross_correlation(
                df,
                x_name,
                y_name,
                self.max_lag_spin.value()
            )

        self.status_label.setText(
            f"{x_name} vs {y_name} | "
            f"Lag {lag} | "
            f"Valid observations: {len(x)}"
        )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = StudyDashboard()
    window.show()

    sys.exit(app.exec())