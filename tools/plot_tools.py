from langchain.tools import Tool
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import numpy as np

# ---------- LINE PLOT TOOL ----------
def plot_data(input_str: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] PlotTool invoked with input: {input_str[:200]}
          \n-------------------------------------------------\n""")

    try:
        # Expected format: "x=column1; y=column2; data=CSV string"
        parts = dict(part.strip().split("=", 1) for part in input_str.split(";"))
        x_col = parts["x"]
        y_col = parts["y"]
        csv_data = parts["data"]

        df = pd.read_csv(StringIO(csv_data))

        filename = "plot.png"
        plt.figure()
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return f"Line plot saved as '{filename}' in {os.getcwd()}"

    except Exception as e:
        return f"[PlotTool Error] {str(e)}"

plot_tool = Tool(
    name="PlotTool",
    func=plot_data,
    description=(
        "Use this to create a line plot from CSV data. "
        "Input should be in the format:\n"
        "\"x=column1; y=column2; data=CSV string\"\n"
        "It saves the output as 'plot.png' in the current directory."
    )
)

# ---------- SCATTER PLOT TOOL ----------
def scatterplot_data(input_str: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] ScatterPlotTool invoked with input: {input_str[:200]}
          \n-------------------------------------------------\n""")

    try:
        # Expected format: "x=column1; y=column2; data=CSV string"
        parts = dict(part.strip().split("=", 1) for part in input_str.split(";"))
        x_col = parts["x"]
        y_col = parts["y"]
        csv_data = parts["data"]

        df = pd.read_csv(StringIO(csv_data))

        filename = "scatterplot.png"
        plt.figure()
        plt.scatter(df[x_col], df[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return f"Scatter plot saved as '{filename}' in {os.getcwd()}"

    except Exception as e:
        return f"[ScatterPlotTool Error] {str(e)}"

scatterplot_tool = Tool(
    name="ScatterPlotTool",
    func=scatterplot_data,
    description=(
        "Use this to create a scatter plot from CSV data. "
        "Input should be in the format:\n"
        "\"x=column1; y=column2; data=CSV string\"\n"
        "It saves the output as 'scatterplot.png' in the current directory."
    )
)

# ---------- SCATTER PLOT WITH REGRESSION TOOL ----------
def scatterplot_regression_data(input_str: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] ScatterPlotRegressionTool invoked with input: {input_str[:200]}
          \n-------------------------------------------------\n""")

    try:
        # Expected format: "x=column1; y=column2; data=CSV string"
        parts = dict(part.strip().split("=", 1) for part in input_str.split(";"))
        x_col = parts["x"]
        y_col = parts["y"]
        csv_data = parts["data"]

        df = pd.read_csv(StringIO(csv_data))

        X = df[[x_col]].values
        y = df[y_col].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        # Regression stats
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        filename = "scatterplotregression.png"
        plt.figure()
        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col} with Regression Line')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.close()

        result = (
            f"Scatter plot with regression line saved as '{filename}' in {os.getcwd()}\n\n"
            f"--- Regression Summary ---\n"
            f"Slope: {slope:.4f}\n"
            f"Intercept: {intercept:.4f}\n"
            f"R² Score: {r2:.4f}\n"
            f"Mean Squared Error: {mse:.4f}"
        )

        return result

    except Exception as e:
        return f"[ScatterPlotRegressionTool Error] {str(e)}"

scatterplot_regression_tool = Tool(
    name="ScatterPlotRegressionTool",
    func=scatterplot_regression_data,
    description=(
        "Use this to create a scatter plot from CSV data and plot a regression line. "
        "It also returns regression coefficients, R² score, and mean squared error.\n"
        "Input should be in the format:\n"
        "\"x=column1; y=column2; data=CSV string\"\n"
        "The plot is saved as 'scatterplotregression.png' in the current directory."
    )
)

