import pandas as pd
import numpy as np
import os, json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

def scientific_notation_formatter(val, pos):
    """Formatter function for scientific notation."""
    if abs(val) > 1000 or (abs (val) < 0.001 and val != 0):
        return f'{val:.1e}'
    else:
        return f'{val:.2f}'

def load_experimental_data(fname):
    df = pd.read_csv(fname)
    objs = ['abs', 'em', 'E']
    obj_vals = df[objs].values
    IDs = df['ID'].values.astype(int)
    return IDs, obj_vals, objs

def load_computational_data(IDs, root_path='./tddft/'):
    data = []
    for id in IDs:
        if os.path.exists(root_path + f'{id}/EXC_ABS.csv'):
            # We have the data
            abs_df = pd.read_csv(root_path + f'{id}/GS_ABS.csv', index_col=0)
            exc_df = pd.read_csv(root_path + f'{id}/EXC_ABS.csv', index_col=0)
            gs_HL = json.load(open(root_path + f'{id}/gs_energies.json'))
            exc_HL = json.load(open(root_path + f'{id}/exc_energies.json'))
            data.append({'ID': id, 'abs': abs_df, 'em': exc_df, 'gs_HL': gs_HL, 'exc_HL': exc_HL})
    return data

def get_data(exp_data, comp_data, IDs):
    max_abs_data = []
    max_em_data = []
    E_data = []
    gs_HL_data = []
    exc_HL_data = []
    for comp in comp_data:
        
        # get experimental max abs
        max_abs_lamda = exp_data[IDs == comp['ID']][0][0]
        max_em_lambda = exp_data[IDs == comp['ID']][0][1]
        E = exp_data[IDs == comp['ID']][0][2]
        
        # get computational HL-Gap
        gs_HL = comp['gs_HL']['gap']
        exc_HL = comp['exc_HL']['gap']
        
        max_abs_data.append(max_abs_lamda)
        max_em_data.append(max_em_lambda)
        E_data.append(E)
        
        gs_HL_data.append(gs_HL)
        exc_HL_data.append(exc_HL)
        
    # Standard scale the experimental data
    max_abs_data = StandardScaler().fit_transform(np.array(max_abs_data).reshape(-1, 1)).flatten()
    max_em_data = StandardScaler().fit_transform(np.array(max_em_data).reshape(-1, 1)).flatten()
    gs_HL_data = MinMaxScaler().fit_transform(np.array(gs_HL_data).reshape(-1, 1)).flatten()
    exc_HL_data = MinMaxScaler().fit_transform(np.array(exc_HL_data).reshape(-1, 1)).flatten()
        
    max_abs_dict = {'name': 'MAX_ABS_LAMBDA', 'data': max_abs_data}
    max_em_dict = {'name': 'MAX_EM_LAMBDA', 'data': max_em_data}
    # E_dict = {'name': 'E', 'data': E_data}
    
    gs_HL_dict = {'name': 'GS_HL_GAP_eV', 'data': gs_HL_data}
    exc_HL_dict = {'name': 'EXC_HL_GAP_eV', 'data': exc_HL_data}
    
    # All combinations of correlations to plot
    combinations = [
        (max_abs_dict, gs_HL_dict, 'green'),
        (max_abs_dict, exc_HL_dict, 'gray'),
        (max_em_dict, gs_HL_dict, 'blue'),
        (max_em_dict, exc_HL_dict, 'red'),
        # (E_dict, gs_HL_dict, 'orange'),
        # (E_dict, exc_HL_dict, 'purple')
    ]
    return combinations

def find_outliers(y_pred, y_data, thresh=2.5):
    residuals = y_data - y_pred
    standardized_residuals = residuals / np.std(residuals)
    return np.where(np.abs(standardized_residuals) > thresh)[0]
    
def fit_model(x_data, y_data, outlier_indices=None):
    # Fit a linear regression model
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1, 1)
    if len(y_data.shape) == 1:
        y_data = y_data.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    r2 = r2_score(y_data, y_pred)
    slope = model.coef_[0, 0]
    intercept = model.intercept_[0]
    
    # Find outliers
    outliers_ = find_outliers(y_pred, y_data)
    if outlier_indices is None:
        outlier_indices = outliers_
    else:
        outlier_indices = np.concatenate([outlier_indices, outliers_])
    
    # If we have any outliers, remove them and fit a new model
    if len(outliers_) > 0:
        outlier_mask = np.ones(len(x_data), dtype=bool)
        outlier_mask[outliers_] = False
        return fit_model(x_data[outlier_mask], y_data[outlier_mask], outlier_indices=outlier_indices)
    else:
        return y_pred, r2, slope, intercept, x_data, outlier_indices

def plot_combinations(combinations, axs):
    # Loop through combinations and plot
    for idx, (x_dict, y_dict, c) in enumerate(combinations):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        x_data = np.array(x_dict['data']).reshape(-1, 1)
        y_data = np.array(y_dict['data']).reshape(-1, 1)

        # Fit the model
        _, r2, slope, intercept, _, _ = fit_model(x_data, y_data)
        # Take the model and get predictions
        y_pred = slope * x_data + intercept
        # Check for outliers
        outlier_indices = find_outliers(y_pred, y_data, thresh=2.0)
        outlier_mask = np.ones(len(x_data), dtype=bool)
        outlier_mask[outlier_indices] = False
        # Plot the regression line without outliers
        x_data_no_outliers = x_data[outlier_mask]
        y_pred_no_outliers = slope * x_data_no_outliers + intercept
        ax.plot(x_data_no_outliers, y_pred_no_outliers, color='black', linewidth=1)
        # Plot the outliers
        if len(outlier_indices) > 0:
            ax.scatter(x_data[outlier_indices], y_data[outlier_indices], s=100, color='k', edgecolor='k', linewidth=1.0)
        
        # Scatter plot
        ax.scatter(x_data, y_data, s=75, color=c)
        ax.set_xlabel(x_dict['name'], labelpad=15)
        ax.set_ylabel(y_dict['name'], labelpad=15)
        ax.set_title(f'{x_dict["name"]} vs {y_dict["name"]}')


        # Circle outliers
        if len(outlier_indices) > 0:
            ax.scatter(x_data[outlier_indices], y_data[outlier_indices], s=100, color='k', edgecolor='k', linewidth=1.0)

        # Prepare the equation and R^2 text
        slope_text = str(round(slope, 4))
        intercept_text = str(round(intercept, 4))
        equation_text = f'$y = {slope_text} * x + {intercept_text}$\n$R^2 = {r2:.3f}$'

        # Display the equation and R^2 value with a box
        props = dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black', linewidth=2.0)
        if r2 > 0.75:
            props['edgecolor'] = 'limegreen'
            props['linewidth'] = 2.0
        elif r2 < 0.1:
            props['edgecolor'] = 'red'
            props['linewidth'] = 2.0
        ax.text(0.25, -0.2, equation_text, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        # Set the formatter for x and y axis ticks
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation_formatter))
        
        # set xtick labels
        ax.set_xticks(np.arange(-2.5, 3, 0.5))
        ax.set_yticks(np.arange(0, 1.1, 0.1))

if __name__ == "__main__":
    exp_path = './data/experimental_data.csv'
    IDs, exp_data, obj_names = load_experimental_data(exp_path)
    comp_data = load_computational_data(IDs)
    combinations = get_data(exp_data, comp_data, IDs)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(22, 12))
    plot_combinations(combinations, axs)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('./data/all_correlations.png')
    print("Plots saved to './data/all_correlations.png'")