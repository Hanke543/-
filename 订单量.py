import numpy as np
def calculate_weighted_sales(sales_data, weights):
    weighted_sales = []
    for i in range(len(sales_data)):
        weight = weights[i % len(weights)]  # 循环使用权重
        weighted_sale = sales_data[i] * weight
        weighted_sales.append(weighted_sale)
    return weighted_sales

def calculate_average_weighted_sales(weighted_sales, n):
    if len(weighted_sales) < n:
        return sum(weighted_sales) / len(weighted_sales) if weighted_sales else 0
    return sum(weighted_sales[-n:]) / n

def calculate_adjusted_sales(sales_data, adjustment_factors):
    adjusted_sales = []
    for i in range(len(sales_data)):
        adjustment_factor = adjustment_factors[i % len(adjustment_factors)]  # 循环使用调整因子
        adjusted_sale = sales_data[i] * adjustment_factor
        adjusted_sales.append(adjusted_sale)
    return adjusted_sales

def calculate_average_adjusted_sales(adjusted_sales, years, week):
    total = 0
    count = 0
    for i in range(years):
        index = week - i * 52
        if index >= 0:
            total += adjusted_sales[index]
            count += 1
    return total / count if count > 0 else 0

def calculate_seasonal_coefficients(adjusted_sales, years):
    if len(adjusted_sales) < 52:
        raise ValueError("至少需要一年的调整后销售数据来计算季节性系数")
    
    seasonal_coefficients = []
    for week in range(52):
        average_adjusted_sale = calculate_average_adjusted_sales(adjusted_sales, years, week + 1)
        seasonal_coefficient = adjusted_sales[week] / average_adjusted_sale if average_adjusted_sale != 0 else 1
        seasonal_coefficients.append(seasonal_coefficient)
    return seasonal_coefficients

def calculate_promotion_factors(promotion_data):
    return promotion_data

def calculate_price_factor(price_data):
    return sum(price_data) / len(price_data) if price_data else 1

def calculate_other_factor(other_data):
    return sum(other_data) / len(other_data) if other_data else 1


def forecast_sales(weighted_sales, seasonal_coefficient, promotion_factors, price_factor, other_factor, n, m):
    average_weighted_sale = calculate_average_weighted_sales(weighted_sales, n)
    seasonal_coefficient = min(seasonal_coefficient, 2)  # 限制季节性系数最大为2
    promotion_factor = min(promotion_factors.get('current_promotion', 1), 2)  # 限制促销因子最大为2

    # 限制price_factor和other_factor
    price_factor = min(price_factor, 1.5)
    other_factor = min(other_factor, 1.5)

    forecast = average_weighted_sale * seasonal_coefficient * promotion_factor * price_factor * other_factor
    return forecast

def moving_average_forecast(sales, window):
    return np.convolve(sales, np.ones(window)/window, 'valid')



def main():
    # 示例数据
    sales_data = [1000, 2000, 1100, 1500, 2000, 1600, 1700, 1000, 1900, 2000] * 6  # 历史周销售数据
    weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # 权重,越近的数据权重越高
    adjustment_factors = [1.0, 1.1, 0.9, 1.2, 1.3, 0.8, 1.1, 0.95, 1.05, 1.15]  # 调整因子
    promotion_data = {'current_promotion': 1.3}  # 促销因子
    price_data = [100, 110, 120, 100, 90, 95]  # 价格数据
    other_data = [1.0, 0.9, 1.1, 1.05, 0.95]  # 其他因素数据
    n = 4  # 计算过去n周的加权平均销售
    m = 2  # 计算过去m年的调整后平均销售
    forecast_weeks = 5  # 预测未来5周的销售

    weighted_sales = calculate_weighted_sales(sales_data, weights)
    adjusted_sales = calculate_adjusted_sales(sales_data, adjustment_factors)
    seasonal_coefficients = calculate_seasonal_coefficients(adjusted_sales, m)
    # 归一化季节性系数
    seasonal_coefficients = [coef / np.mean(seasonal_coefficients) for coef in seasonal_coefficients]  
    price_factor = calculate_price_factor(price_data)
    promotion_factor = calculate_promotion_factors(promotion_data)
    other_factor = calculate_other_factor(other_data)

    # 使用正确的数据和窗口大小
    predicted_sales = moving_average_forecast(sales_data, window=3)

    forecasts = []
    for i in range(forecast_weeks):
        seasonal_index = (len(sales_data) + i) % len(seasonal_coefficients)
        seasonal_coefficient = min(seasonal_coefficients[seasonal_index], 2)
        forecast = forecast_sales(weighted_sales, seasonal_coefficient, promotion_factor, price_factor, other_factor, n, m)
        forecasts.append(forecast)

        # 用平滑的预测值更新weighted_sales
        smoothed_forecast = 0.2 * forecast + 0.8 * sales_data[-1]
        weighted_sales.append(smoothed_forecast)
        weighted_sales.pop(0)

    

    print("预测的销售数据：", forecasts)

if __name__ == "__main__":
    main()
