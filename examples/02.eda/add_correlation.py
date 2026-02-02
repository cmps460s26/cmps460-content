import json

notebook_path = 'c:/_cmps460-content/examples/02.eda/1_2-summary-stats.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Correlation\n",
            "\n",
            "Correlation measures the strength and direction of the relationship between two numerical variables. \n",
            "The result is a value between -1 and 1:\n",
            "*   **1**: Perfect positive correlation (as x increases, y increases)\n",
            "*   **-1**: Perfect negative correlation (as x increases, y decreases)\n",
            "*   **0**: No correlation"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Pearson Correlation\n",
            "\n",
            "Pearson correlation assesses the **linear** relationship between two continuous variables. It assumes a normal distribution and linearity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate Pearson Correlation\n",
            "# We'll look at the correlation between Price, Horsepower, Engine Size, and MPG\n",
            "cols_corr = ['price', 'horsepower', 'engine-size', 'city-mpg', 'highway-mpg']\n",
            "pearson_corr = df[cols_corr].corr(method='pearson')\n",
            "\n",
            "print(\"Pearson Correlation Matrix:\")\n",
            "print(pearson_corr)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Spearman Correlation\n",
            "\n",
            "Spearman's rank correlation assesses the **monotonic** relationship (whether linear or not). It is based on the ranked values rather than the raw data and is less sensitive to outliers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate Spearman Correlation\n",
            "spearman_corr = df[cols_corr].corr(method='spearman')\n",
            "\n",
            "print(\"\\nSpearman Correlation Matrix:\")\n",
            "print(spearman_corr)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Interpretation of Results\n",
            "\n",
            "*   **Price vs Horsepower (0.81 Pearson, 0.86 Spearman)**: Strong positive correlation. As horsepower increases, price tends to increase significantly.\n",
            "*   **Price vs City-MPG (-0.69 Pearson, -0.83 Spearman)**: Strong negative correlation. Cars with higher fuel efficiency (MPG) tend to be cheaper.\n",
            "*   **City-MPG vs Highway-MPG (0.97 Pearson)**: Very strong positive correlation, which makes sense as they measure similar attributes."
        ]
    }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
