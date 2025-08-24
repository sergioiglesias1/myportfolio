# Sergio Iglesias - Portfolio: Beta-Convergence Analysis

## Project Overview

This project investigates Beta-Convergence across Emerging and Developed economies over four distinct periods:

- 2004–2008
- 2009–2013  
- 2014–2018
- 2019–2024

Using Python (Pandas, Matplotlib, and Scikit-learn), I prepared the data, visualized growth trends, and ran regression models to determine convergence patterns. The goal is to understand whether countries with lower initial GDP per capita are catching up with wealthier nations and to analyze periods of divergence.

## Table of Contents

- [Requirements](#requirements)
- [Data Acquisition](#data-acquisition)
- [Beta-Convergence Implementation](#beta-convergence-implementation)
- [Visualizations](#visualizations)
- [Interpretation and Limitations](#interpretation-and-limitations)
- [How to Run](#how-to-run)
- [Contact](#contact)
- [Credits](#credits)

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - requests (for API data access)

Install dependencies with:

```
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

## Data Acquisition

The analysis uses GDP per capita data from the World Development Indicators (WDI) database. Data is retrieved through the World Bank API or directly from their databank.

Key variables:
- GDP per capita (constant 2015 US$)
- Country classification (Emerging vs. Developed economies)
- Time period: 2004-2024

## Beta-Convergence Implementation

Using scikit-learn's LinearRegression, I estimate convergence coefficients for each period:

```
# Example code structure
from sklearn.linear_model import LinearRegression

# Prepare data for a specific period
X = initial_gdp_values.reshape(-1, 1)
y = growth_rates

# Run regression
model = LinearRegression().fit(X, y)

# Extract beta coefficient
beta_coefficient = model.coef_[0]
```

The analysis includes both absolute convergence (all countries) and conditional convergence (within country groups).

## Visualizations

The project generates several visual components:

- Growth vs. Initial GDP scatter plots for each period
- Regression lines showing convergence/divergence trends
- Comparison charts between Emerging and Developed economies
- Time series of GDP per capita for selected countries
- Coefficient plots showing how beta values change across periods

## Interpretation and Limitations

Key findings:
- Identification of periods with strong convergence versus divergence
- Differences in convergence patterns between Emerging and Developed economies
- Impact of global economic events on convergence trends

Limitations:
- Data availability and quality for some economies
- Simplified model that doesn't capture all convergence determinants
- Potential omitted variable bias in the regression analysis

## How to Run

1. Clone this repository
2. Install required packages
3. Run the Jupyter Notebook `beta_convergence_analysis.ipynb`
4. Explore the visualizations and results
5. For the web portfolio, open `index.html` in your browser

## Contact

- **Email**: sergioiglesiaslopez03@gmail.com
- **Location**: Santander, Cantabria, 39012
- **LinkedIn**: https://www.linkedin.com/in/sergio-iglesias-179aa323b/
- **GitHub**: https://github.com/sergioiglesias1

## Credits

### Demo Images
Unsplash (unsplash.com) – CC0 (public domain) images (not included).

### Icons
Font Awesome (fontawesome.io).

### Other Tools
- jQuery (jquery.com)
- Scrollex (github.com/ajlkn/jquery.scrollex)
- Responsive Tools (github.com/ajlkn/responsive-tools)

### Data Source
World Development Indicators (WDI). More info: https://databank.worldbank.org

### Analysis
Developed by Sergio Iglesias using Python libraries.

### Original Code Adaptation
The original Massively code has been adapted to include my portfolio and project content.

### License Details
For the full license details, visit html5up.net/license.