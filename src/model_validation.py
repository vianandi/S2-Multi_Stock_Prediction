import pandas as pd
import numpy as np

def load_actual_data(file_path='dataset/pembanding/datasahampembanding.csv'):
    """
    Load actual prices dari data IDX
    
    Returns:
        dict {stock_code: actual_price}
    """
    df = pd.read_csv(file_path)
    
    # Create dictionary: stock_code -> actual_price
    actual_dict = {}
    for _, row in df.iterrows():
        code = row['Kode Saham'].strip()
        actual_price = row['Penutupan']
        actual_dict[code] = actual_price
    
    return actual_dict

def calculate_error_metrics(predicted, actual):
    """
    Calculate various error metrics
    
    Returns:
        dict with MAE, RMSE, MAPE, accuracy
    """
    error = predicted - actual
    abs_error = abs(error)
    pct_error = abs_error / actual * 100
    
    metrics = {
        'error': error,
        'abs_error': abs_error,
        'pct_error': pct_error,
        'squared_error': error ** 2
    }
    
    return metrics

def validate_predictions(df_comparison, actual_data):
    """
    Validate model predictions against actual data
    
    Parameters:
    -----------
    df_comparison : DataFrame
        Results from model comparison with columns:
        stock, arima_forecast, lstm_forecast, ensemble_forecast, 
        stacked_forecast, weighted_forecast
    actual_data : dict
        Dictionary {stock_code: actual_price}
    
    Returns:
    --------
    DataFrame with validation results and metrics
    """
    validation_results = []
    
    for _, row in df_comparison.iterrows():
        stock = row['stock']
        
        # Skip if no actual data available
        if stock not in actual_data:
            continue
        
        actual_price = actual_data[stock]
        current_price = row['current_price']
        
        result = {
            'stock': stock,
            'current': current_price,
            'actual': actual_price,
            'actual_change%': (actual_price - current_price) / current_price * 100
        }
        
        # Validate all available models
        for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
            forecast_col = f'{model}_forecast'
            if forecast_col in df_comparison.columns and pd.notna(row[forecast_col]):
                pred = row[forecast_col]
                metrics = calculate_error_metrics(pred, actual_price)
                result[f'{model}_pred'] = pred
                result[f'{model}_error'] = metrics['pct_error']  # Use percentage error
                result[f'{model}_abs_error'] = metrics['abs_error']
                result[f'{model}_direction'] = 'correct' if (pred > current_price) == (actual_price > current_price) else 'wrong'
        
        validation_results.append(result)
    
    df_validation = pd.DataFrame(validation_results)
    return df_validation

def calculate_aggregate_metrics(df_validation):
    """
    Calculate aggregate performance metrics for each model
    """
    metrics_summary = {}
    
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        error_col = f'{model}_abs_error'
        pct_error_col = f'{model}_error'
        pred_col = f'{model}_pred'
        
        if error_col in df_validation.columns:
            valid_data = df_validation[df_validation[error_col].notna()]
            
            if len(valid_data) > 0:
                # Calculate R² (coefficient of determination)
                actuals = valid_data['actual'].values
                predictions = valid_data[pred_col].values
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - actuals.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics_summary[model] = {
                    'count': len(valid_data),
                    'MAE': valid_data[error_col].mean(),
                    'RMSE': np.sqrt((valid_data[error_col] ** 2).mean()),
                    'MAPE': valid_data[pct_error_col].mean(),
                    'R2': r2,
                    'min_error': valid_data[error_col].min(),
                    'max_error': valid_data[error_col].max(),
                    'std_error': valid_data[error_col].std()
                }
    
    return metrics_summary

def determine_best_model(metrics_summary):
    """
    Determine which model performs best based on MAPE
    """
    if not metrics_summary:
        return None
    
    best_model = min(metrics_summary.items(), 
                     key=lambda x: x[1]['MAPE'])
    
    return best_model[0], best_model[1]

def calculate_directional_accuracy(df_validation):
    """
    Calculate accuracy of predicting price direction (up/down)
    """
    accuracy = {}
    
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        direction_col = f'{model}_direction'
        
        if direction_col in df_validation.columns:
            valid_data = df_validation[df_validation[direction_col].notna()]
            
            if len(valid_data) > 0:
                correct = (valid_data[direction_col] == 'correct').sum()
                total = len(valid_data)
                
                accuracy[model] = {
                    'correct': correct,
                    'total': total,
                    'accuracy_pct': correct / total * 100
                }
    
    return accuracy

def generate_validation_report(df_validation, metrics_summary, directional_accuracy, output_path):
    """
    Generate comprehensive validation report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL VALIDATION REPORT\n")
        f.write("Prediction vs Actual (IDX Data - 9 January 2023)\n")
        f.write("="*80 + "\n\n")
        
        # Overall Summary
        f.write("VALIDATION SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"  Total Stocks:      {len(df_validation)}\n")
        f.write(f"  Validation Date:   9 January 2023\n")
        f.write(f"  Data Source:       IDX (Indonesia Stock Exchange)\n\n")
        
        # Best Model
        if metrics_summary:
            best_model, best_metrics = determine_best_model(metrics_summary)
            f.write("🏆 BEST PERFORMING MODEL\n")
            f.write("-"*80 + "\n")
            f.write(f"  Winner: {best_model}\n")
            f.write(f"  MAPE:   {best_metrics['MAPE']:.2f}%\n")
            f.write(f"  MAE:    Rp {best_metrics['MAE']:,.2f}\n")
            f.write(f"  RMSE:   Rp {best_metrics['RMSE']:,.2f}\n\n")
        
        # Error Metrics Comparison
        f.write("ERROR METRICS COMPARISON\n")
        f.write("-"*80 + "\n")
        for model, metrics in metrics_summary.items():
            f.write(f"  {model.upper():10s}:\n")
            f.write(f"    MAPE: {metrics['MAPE']:6.2f}%\n")
            f.write(f"    MAE:  Rp {metrics['MAE']:10,.2f}\n")
            f.write(f"    RMSE: Rp {metrics['RMSE']:10,.2f}\n")
            if 'R2' in metrics:
                f.write(f"    R²:   {metrics['R2']:8.4f}\n")
            f.write("\n")
        
        # Directional Accuracy
        f.write("DIRECTIONAL ACCURACY (Predicted Direction vs Actual)\n")
        f.write("-"*80 + "\n")
        for model, acc in directional_accuracy.items():
            f.write(f"  {model.upper():10s}: {acc['accuracy_pct']:.2f}% ")
            f.write(f"({acc['correct']}/{acc['total']} correct predictions)\n")
        f.write("\n")
        
        # Detailed per-stock validation
        f.write("DETAILED VALIDATION TABLE\n")
        f.write("-"*80 + "\n")
        
        # Table header
        header = f"{'Stock':<6} {'Current':<8} {'Actual':<8} {'ARIMA':<8} {'LSTM':<8} {'Ensemble':<8} {'Best':<8}\n"
        f.write(header)
        f.write("-"*80 + "\n")
        
        # Table rows
        for _, row in df_validation.iterrows():
            # Determine best model for this stock
            errors = {
                'ARIMA': abs(row.get('arima_error', 999)),
                'LSTM': abs(row.get('lstm_error', 999)),
                'Ensemble': abs(row.get('ensemble_error', 999))
            }
            best_for_stock = min(errors.items(), key=lambda x: x[1])[0]
            
            line = f"{row['stock']:<6} "
            line += f"{row['current']:<8.0f} "
            line += f"{row['actual']:<8.0f} "
            line += f"{row.get('arima_pred', 0):<8.0f} "
            line += f"{row.get('lstm_pred', 0):<8.0f} "
            line += f"{row.get('ensemble_pred', 0):<8.0f} "
            line += f"{best_for_stock:<8}\n"
            f.write(line)
        
        f.write("\n" + "="*80 + "\n")
        f.write("Legend:\n")
        f.write("  MAPE  = Mean Absolute Percentage Error (lower is better)\n")
        f.write("  MAE   = Mean Absolute Error (lower is better)\n")
        f.write("  RMSE  = Root Mean Squared Error (lower is better)\n")
        f.write("  R²    = Coefficient of Determination (higher is better, max 1.0)\n")
        f.write("  Best  = Model with lowest error for each stock\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Validation report saved: {output_path}")