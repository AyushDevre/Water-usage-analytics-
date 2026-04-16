import pandas as pd
import random

industries = {
    "Textile": {"base": 500000},
    "Data Center": {"base": 400000},
    "Manufacturing": {"base": 500000},
    "Agriculture": {"base": 800000},
    "Steel": {"base": 600000},
    "Pharma": {"base": 300000}
}

locations = ["India", "USA", "Germany", "Brazil", "China", "UK"]

data = []

for industry, vals in industries.items():
    base = vals["base"]

    for location in locations:
        growth = base

        for year in range(2015, 2024):
            growth = growth * random.uniform(1.03, 1.08)  # yearly increase

            water_usage = int(growth + random.randint(-20000, 20000))
            production_units = random.randint(5000, 30000)
            energy_consumption = random.randint(10000, 100000)

            data.append([
                industry,
                location,
                year,
                water_usage,
                production_units,
                energy_consumption
            ])

df = pd.DataFrame(data, columns=[
    "industry",
    "location",
    "year",
    "water_usage",
    "production_units",
    "energy_consumption"
])

df.to_csv("data/data.csv", index=False)

print("✅ Realistic trending dataset generated!")