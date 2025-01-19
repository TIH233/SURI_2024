import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the CSV file
file_path = 'RestaurantDataVets_All_2to5 (1).csv'
df = pd.read_csv(file_path)

# Extract the sales column
sales_data = df['2to5']

'''
# Display basic information about the sales data
print(sales_data.describe())

# Plot a histogram of the sales data
plt.figure(figsize=(10, 6))
plt.hist(sales_data, bins=30, edgecolor='black')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Check for normality
_, p_value = stats.normaltest(sales_data)
print(f"p-value for normality test: {p_value}")

# Right skewed and high outliers, which indicates factors that lead to extreme values.
#----------------------------------------------------------------------------------------------------------------
# List of distributions to try
distributions = [
    stats.gamma,
    stats.lognorm,
    stats.weibull_min,
    stats.exponweib,
    stats.burr12
]

# Fit distributions
results = []
for dist in distributions:
    params = dist.fit(sales_data)
    D, p_value = stats.kstest(sales_data, dist.name, args=params)
    results.append((dist.name, D, p_value))

# Sort results by test statistic (lower is better fit)
results.sort(key=lambda x: x[1])

# Print results
for name, D, p_value in results:
    print(f"{name:12}: D = {D:.4f}, p-value = {p_value:.4f}")

# Plot the best fit distribution
best_dist = getattr(stats, results[0][0])
best_params = best_dist.fit(sales_data)

x = np.linspace(min(sales_data), max(sales_data), 100)
pdf = best_dist.pdf(x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])

plt.figure(figsize=(10, 6))
plt.hist(sales_data, bins=30, density=True, alpha=0.7, label='Data')
plt.plot(x, pdf, 'r-', label=f'Best Fit ({results[0][0]})')
plt.title('Sales Data with Best Fit Distribution')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.legend()
plt.show()

# Print parameters of the best fit distribution
print(f"\nBest fit distribution: {results[0][0]}")
print("Parameters:", best_params)

#best is gamma, with p-value still too small
'''
#--------------------------------------------------------------------------------------------------
from sklearn.mixture import GaussianMixture

# Fit a mixture of two Gaussian distributions
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(sales_data.values.reshape(-1, 1))  # Convert to numpy array and reshape

# Generate points from the GMM
x = np.linspace(min(sales_data), max(sales_data), 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)

# Plot the results
plt.figure(figsize=(10, 6))
plt.hist(sales_data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x, pdf, 'r-', label='Gaussian Mixture Model')
plt.title('Sales Data with Gaussian Mixture Model')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.legend()
plt.show()

# Print the parameters of the mixture model
for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
    print(f'Component {i+1}: mean = {mean[0]:.2f}, std = {np.sqrt(cov[0][0]):.2f}')
print(f'Mixture weights: {gmm.weights_}')

# Calculate CV² for each component and the overall distribution
def calculate_cv_squared(mean, variance):
    return variance / (mean ** 2)

overall_mean = np.sum(gmm.weights_ * gmm.means_[:, 0])
overall_variance = np.sum(gmm.weights_ * (gmm.covariances_[:, 0, 0] + gmm.means_[:, 0]**2)) - overall_mean**2

for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
    cv_squared = calculate_cv_squared(mean[0], cov[0][0])
    print(f'Component {i+1} CV² = {cv_squared:.4f}')

overall_cv_squared = calculate_cv_squared(overall_mean, overall_variance)
print(f'Overall CV² = {overall_cv_squared:.4f}')

# Function to generate samples from the GMM
def sample_gmm(n_samples):
    components = np.random.choice(2, size=n_samples, p=gmm.weights_)
    samples = np.array([
        np.random.normal(gmm.means_[c, 0], np.sqrt(gmm.covariances_[c, 0, 0]))
        for c in components
    ])
    return samples

# Generate samples for Monte Carlo simulation
mc_samples = sample_gmm(10000)

# Plot histogram of Monte Carlo samples
plt.figure(figsize=(10, 6))
plt.hist(mc_samples, bins=50, density=True, alpha=0.7, label='Monte Carlo Samples')
plt.hist(sales_data, bins=50, density=True, alpha=0.7, label='Original Data')
plt.title('Comparison of Original Data and Monte Carlo Samples')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.legend()
plt.show()

# Calculate CV² for Monte Carlo samples
mc_mean = np.mean(mc_samples)
mc_variance = np.var(mc_samples)
mc_cv_squared = calculate_cv_squared(mc_mean, mc_variance)
print(f'Monte Carlo Samples CV² = {mc_cv_squared:.4f}')

# Provide good fit and reflects bimodel nature of the distribution,
# but cannot fully capture the small peak in lower sales days
# However, the potential time dependent pattern is not discovered and the points are independent
#which will influence the sequencing of distribution
#----------------------------------------------------------------------------------------------------------
'''
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Extract the sales column and convert to numpy array
sales_data = df['2to5'].values

# Create a numeric index
numeric_index = np.arange(len(sales_data))

# Time series plot
plt.figure(figsize=(15, 6))
plt.plot(numeric_index, sales_data)
plt.title('Sales Over Time')
plt.xlabel('Time Index')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig('sales_over_time.png')
plt.close()

# Simple decomposition
def simple_decompose(data, period):
    trend = np.convolve(data, np.ones(period)/period, mode='same')
    seasonal = np.zeros_like(data)
    for i in range(period):
        seasonal[i::period] = np.mean(data[i::period] - trend[i::period])
    residual = data - trend - seasonal
    return trend, seasonal, residual

trend, seasonal, residual = simple_decompose(sales_data, 7)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
ax1.plot(numeric_index, sales_data)
ax1.set_title('Observed')
ax2.plot(numeric_index, trend)
ax2.set_title('Trend')
ax3.plot(numeric_index, seasonal)
ax3.set_title('Seasonal')
ax4.plot(numeric_index, residual)
ax4.set_title('Residual')
plt.tight_layout()
plt.savefig('simple_decomposition.png')
plt.close()

# Autocorrelation and Partial Autocorrelation plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(sales_data, ax=ax1, lags=50)
plot_pacf(sales_data, ax=ax2, lags=50)
plt.tight_layout()
plt.savefig('acf_pacf.png')
plt.close()

# Box plot for day of week patterns
day_of_week = numeric_index % 7
plt.figure(figsize=(10, 6))
sns.boxplot(x=day_of_week, y=sales_data)
plt.title('Sales Distribution by Day of Week')
plt.xlabel('Day of Week (0-6)')
plt.ylabel('Sales')
plt.savefig('sales_by_dayofweek.png')
plt.close()

# Monthly pattern (assuming 30 days per month)
month = numeric_index // 30 + 1
plt.figure(figsize=(12, 6))
sns.boxplot(x=month, y=sales_data)
plt.title('Sales Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.savefig('sales_by_month.png')
plt.close()

print("All graphs have been saved as PNG files.")
#-----------------------------------------------------------------------------------------------------------------------
#The ACF, PACF shows that weekly seasonality and trend
# To reach the final outcome,
# We will try to decompose the data factors (seasonality, residual, trend) with time series models
# Then generate data point for simulation with built model
# At last test the goodness of fit
'''

# However, the simulation should follow the baseline of random, which should not follow specific pattern to simulate a stocastic process