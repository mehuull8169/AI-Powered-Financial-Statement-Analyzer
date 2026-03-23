# financial_tools.py

def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """Calculates the current ratio."""
    if current_liabilities == 0:
        return float('inf')  # Handle division by zero
    return current_assets / current_liabilities

def calculate_debt_to_equity_ratio(total_liabilities: float, shareholders_equity: float) -> float:
    """Calculates the debt-to-equity ratio."""
    if shareholders_equity == 0:
        return float('inf')
    return total_liabilities / shareholders_equity

# ... Add functions for ALL the required ratios from the document ...
def calculate_gross_margin_ratio(gross_profit: float, net_sales: float) -> float:
    if net_sales == 0:
        return 0.0
    return gross_profit / net_sales

def calculate_operating_margin_ratio(operating_income: float, net_sales: float) -> float:
    if net_sales == 0:
        return 0.0
    return operating_income / net_sales

def calculate_return_on_assets_ratio(net_income: float, total_assets: float) -> float:
    if total_assets == 0:
        return 0.0
    return net_income / total_assets

def calculate_return_on_equity_ratio(net_income: float, shareholders_equity: float) -> float:
    if shareholders_equity == 0:
        return 0.0
    return net_income / shareholders_equity

def calculate_asset_turnover_ratio(net_sales: float, average_total_assets: float) -> float:
    if average_total_assets == 0:
        return 0.0
    return net_sales / average_total_assets

def calculate_inventory_turnover_ratio(cost_of_goods_sold: float, average_inventory: float) -> float:
    if average_inventory == 0:
        return 0.0
    return cost_of_goods_sold / average_inventory

def calculate_receivables_turnover_ratio(net_credit_sales: float, average_accounts_receivable: float) -> float:
    if average_accounts_receivable == 0:
        return 0.0
    return net_credit_sales / average_accounts_receivable

def calculate_debt_ratio(total_liabilities: float, total_assets: float) -> float:
    if total_assets == 0:
        return 0.0
    return total_liabilities / total_assets

def calculate_interest_coverage_ratio(operating_income: float, interest_expenses: float) -> float:
    if interest_expenses == 0:
        return float('inf')
    return operating_income / interest_expenses