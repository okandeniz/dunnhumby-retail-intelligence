# Dunnhumby Retail Intelligence

Explore how households shop, which customers may stop visiting, how shoppers use coupons, which products to suggest, and how many units a retailer may sell. The Streamlit dashboard has five tabs. FastAPI serves the saved models and historical results. The dashboard can run offline once the images and project artifacts are in place.

## Start with Docker

From the project root, start Docker Desktop in Linux container mode. Keep the `models/` and `data/processed/` folders beside `compose.yaml`.

```powershell
docker compose up --build -d
docker compose ps
```

Open the [dashboard](http://localhost:8501) or [interactive API documentation](http://localhost:8000/docs). For a quick example, open **Product Demand Forecasting**, select **PRODUCE / TROPICAL FRUIT** and choose bananas with ID **1082185**.

Compose starts two CPU services. The dashboard contacts the API at `http://api:8000` after its health check passes. Only the API mounts the model and processed data folders, both read-only. The images contain the application code and locked dependencies. Raw data and database credentials stay on the host.

```powershell
docker compose logs --tail 50 api streamlit
docker compose down
```

## Dataset and business questions

The project uses *The Complete Journey*. It follows 2,500 households over about two years and includes purchases, products, campaigns and coupon use. The local inventory contains 2,595,732 transaction lines and 92,353 product records. Day numbers in the app refer to this historical dataset. [Dataset source and access](https://www.dunnhumby.com/source-files/)

| Dashboard tab | Question | What you can explore |
| --- | --- | --- |
| CRM & Customer Segmentation | Which groups visit often or spend more? | RFM segments, K-Means groups, customer details and estimated revenue |
| Customer Churn | Who may make no purchase for 90 days? | Customer ID lookup and an editable six-field form |
| Statistical Campaign Analysis | How does coupon use vary across campaigns? | Assignments, redemptions, spending and statistical tests |
| Recommendation Systems | Which new products might interest a household? | Ranked suggestions, similar products and seven methods at K = 10 |
| Product Demand Forecasting | How many units of a product might sell in 30 days? | Future totals, two baselines and three historical test periods |

### Customer value and segmentation

RFM measures days since the last purchase, purchase frequency and total spending. The original rules assign eight RFM labels. K-Means uses purchase frequency, average basket value and recency for the year ending on day 711. It grouped 1,580 households as **Frequent & Recent Shoppers** and 899 as **Less Active Shoppers**. The other 21 had no positive-value purchases in that year.

The CRM pipeline uses BG-NBD to estimate purchases. It combines these estimates with historical basket value to estimate 90-day revenue. The saved result covers 2,466 households. The customer view shows the saved status when an estimate is missing.

### Customer churn

The balanced logistic regression pipeline applies its saved log transform and scaling. It predicts churn at a probability of 0.80 or higher. On the day-622 test snapshot, ROC AUC was 0.893, precision was 0.345, recall was 0.500 and F1 was 0.408. This is an experimental model.

**Customer ID Prediction** looks up a real feature row and scores it with the saved model. The observed outcome has its own label. **Manual Prediction** accepts the yearly basket count, average basket value, days since the last purchase, basket counts for each of the last two 30-day windows, and a yearly purchase flag. The form checks that the counts and activity dates agree.

**Random Fill** draws one of 2,499 test customers. It fills all six fields and lets you edit them before choosing **Predict Churn**. The original test CSV stores labels and predictions. A separate export rebuilds its inputs from transactions before day 622. It checks the customer IDs, classes and probabilities against the original file before saving the inputs. The target and customer ID are excluded from the model features.

Training used snapshots at days 366, 396 and 426. Day 516 was used for validation. The final test used history through day 621 and observed purchases in days 622–711. A customer with no positive-value basket during those 90 days has the observed churn label.

### Campaign analysis

The study covers 7,208 household–campaign assignments and 889 assignments with a redemption. Select a campaign to see its reach, coupon use and spending. The comparison below the selector uses all campaigns.

Coupon use differed by campaign type in the chi-square test (χ² = 107.99, df = 2, p = 3.55 × 10⁻²⁴). The size of the association was small (Cramér’s V = 0.122). The dashboard also shows median spending, Cliff’s delta and Holm-adjusted p-values from the two-sided Mann–Whitney tests. The GEE redemption model accounts for several assignments to the same household. TypeB is its reference group. The groups were observed in shopping records, so these results cannot establish a sales lift caused by a campaign.

### Recommendations

The evaluation used training purchases through day 621. It checked suggestions against new purchases in days 622–711 for 2,266 households. The comparison uses a catalog of 11,603 products. All customer methods exclude products already bought in the model’s training history.

| Method | Customer suggestions | Related products | Score shown |
| --- | --- | --- | --- |
| Popularity Baseline | Yes | — | Number of customers who bought the item |
| User-Based CF | Yes | — | Purchase share weighted by customer similarity |
| Item-Based CF | Yes | Yes | Sum of similarities for customers, cosine similarity for products |
| Content-Based Filtering | Yes | Yes | Sum of content similarities for customers, cosine similarity for products |
| Association Rules | Yes | Yes | Rule confidence |
| Truncated SVD | Yes | — | SVD factor score |
| NMF | Yes | — | NMF factor score |

Product search accepts a name or an ID. Association Rules offers products that start a saved rule. The two neighbor methods use up to ten of a customer's recent training products. Their scores and the factor scores have different scales. NMF training did not confirm convergence.

The saved Truncated SVD run had Precision@10 of 0.0818, Recall@10 of 0.0216, Hit Rate@10 of 0.5159 and Coverage@10 of 0.0617. The dashboard shows all four measures for every method. Coverage measures the share of the catalog reached by the lists.

### Product demand

The original pipeline selected 457 eligible products using data through day 531. Each example uses information available at its forecast origin. The training origins run from day 111 to 501. The validation origins are 531, 561 and 591. The final model used training and validation examples with targets ending by day 621. Its independent test covers 1,371 product-periods at origins 621, 651 and 681.

| Method | MAE (units) | RMSE (units) | WAPE |
| --- | ---: | ---: | ---: |
| XGBoost + Log Target | 24.872 | 53.544 | 27.332% |
| Last 30 Days Baseline | 26.671 | 56.089 | 29.309% |
| Last 60-Day Average Baseline | 25.803 | 53.711 | 28.355% |

The final origin is dataset day 711. The forecast cards show **one total per method for days 712–741 across all stores**. For bananas (1082185), XGBoost predicts **1,241.66 units**. The last 30 days give 1,565 units. Half of the last 60 days gives 1,554 units. In **Next 30-day forecast**, a dashed line connects the last observed total to the XGBoost forecast. A diamond shows the average of the three previous 30-day totals. Switch to **Actual and predicted demand** to compare actual sales with all three methods during the test periods. Actual sales after day 711 are unavailable.

**Forecast by store** allocates the selected product’s 30-day XGBoost total using each store’s share of positive unit sales in the preceding 180 days. The chart shows the ten largest estimates and the table includes every store that sold the product in that window. Across the three historical test periods, the sum of absolute store-product errors was 93.49% of actual units (WAPE), including sales at stores with no units in the lookback window.

## Architecture and files

```text
Browser → Streamlit + Plotly → FastAPI → cached data and saved models
                                 ↑
                    Existing Python inference functions

api/                  FastAPI entry point, data access and Pydantic schemas
  routes/             Five business routers
  services/           Recommendation scoring with saved artifacts
streamlit_app/        Dashboard, HTTP client, views and charts
.streamlit/           Dashboard theme
src/data/             Historical MySQL access and data preparation
src/features/         Customer, product and campaign features
src/models/           Original model code
src/recommendations/  Original product ranking functions
src/statistics/       Campaign tests and models
src/evaluation/       Forecast and recommendation metrics
src/pipelines/        Offline training, exports and inference
models/               Fitted models, catalogs, matrices and metadata
data/processed/       Saved results and small dashboard inputs
data/raw/             Original source files for offline preparation
notebooks/            Seven original analysis notebooks
reports/              Source data checks and analysis figures
sql/ddl/              Historical database setup
tests/                API, saved model and Streamlit checks
```

Python 3.11, pandas, NumPy, SciPy, scikit-learn and XGBoost support the original analysis. The API loads each needed artifact on first use. Streamlit caches successful historical reads for ten minutes. Random samples are drawn on each click. Model training and MySQL access belong to the offline pipelines.

## Required artifacts

Keep matching files from the same project run. A code-only copy may be missing local artifacts because the repository ignores processed data and serialized models. Copy the two artifact folders from the prepared project before starting a new installation. Use trusted joblib files from this project.

| Folder | Files used |
| --- | --- |
| `models/crm/`, `models/segmentation/` | `crm_model_metadata.json`, `segmentation_metadata.json` |
| `data/processed/crm/` | `rfm_segments.csv`, `crm_segment_forecast_summary.csv`, `crm_customer_forecasts.csv` |
| `data/processed/segmentation/` | `cluster_profiles.csv`, `customer_segments.csv` |
| `models/churn/` | `churn_pipeline.joblib`, `churn_model_metadata.json` |
| `data/processed/churn/` | `churn_test_features.csv`, `churn_test_predictions.csv` |
| `data/processed/campaign/` | `campaign_spending.csv`, `campaign_type_summary.csv`, `chi_square_results.json`, `mann_whitney_results.csv`, `gee_redemption_summary.csv`, `campaign_analysis_metadata.json` |
| `models/recommendation/` | Keep the `user_based`, `item_based`, `content_based`, `association_rules`, `svd` and `nmf` subfolders with their catalogs, ID mappings, fitted neighbors, factors, rules and purchase history |
| `data/processed/recommendation/` | `model_comparison_stage4.csv`, `evaluation_metadata.json`, `customer_seed_products.json` |
| `data/processed/forecasting/` | `eligible_product_lookup.csv`, `future_forecast.csv`, `store_sales_180d.csv`, `test_predictions.csv`, `final_test_comparison.csv`. Keep `daily_product_series.csv` for the model test and `forecast_split_summary.csv` for the split details |
| `models/forecasting/` | `metadata.json`. The model test also loads `xgb_log_target.joblib` |

The API serves the saved future forecast to keep product selection fast. The test suite also rebuilds historical features and runs the fitted XGBoost model to check the same banana forecast.

To restore the dashboard inputs from the original `data/raw/transaction_data.csv` and the other files above, run these commands once after installing the environment.

```powershell
uv run --no-sync python -m src.pipelines.export_churn_test_features
uv run --no-sync python -m src.pipelines.export_recommendation_seeds
uv run --no-sync python -m src.pipelines.export_store_sales
```

## Local installation and configuration

Install uv and Python 3.11. From the project root, install the locked packages. The inexact sync keeps other packages already installed for the notebooks.

```powershell
uv sync --locked --inexact
```

Open two terminals in the project root. Start FastAPI in the first terminal and Streamlit in the second.

```powershell
uv run --no-sync uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
uv run --no-sync streamlit run streamlit_app/app.py --server.port 8501
```

Default settings work without an environment file. To change them, copy `.env.example` to `.env` if that file does not already exist.

| Setting | Default | Use |
| --- | --- | --- |
| `API_URL` | `http://127.0.0.1:8000` | API address for local Streamlit. Compose sets the internal address to `http://api:8000` |
| `API_PORT` | `8000` | API port published by Compose |
| `DASHBOARD_PORT` | `8501` | Dashboard port published by Compose |
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | See `.env.example` | Used only by offline pipelines that read the historical database |

The dashboard and tests run without MySQL. For historical database work, use the original SQL files and loaders with your own local database settings. The optional `research` dependency group keeps the original notebook packages. The app images install only their own runtime group. The locked, hash-checked `requirements.txt` covers both app runtimes.

## FastAPI endpoints

The interactive docs show request fields, response types and validation rules. A missing ID returns 404. Invalid input or an unsupported recommendation mode returns 422. A missing required artifact returns 503. The health endpoint checks that the service is running. [Open API docs](http://localhost:8000/docs)

| Method | Path | Result |
| --- | --- | --- |
| GET | `/health`, `/info` | Service health or model and data metadata |
| GET | `/crm/summary`, `/crm/customers/{customer_id}` | KPIs, segments and one customer |
| GET | `/churn/customers/{customer_id}`, `/churn/sample` | Customer prediction or a real test input |
| POST | `/churn/predict` | Prediction from six editable features |
| GET | `/campaigns`, `/campaigns/{campaign_id}`, `/campaigns/study` | Campaign IDs, one campaign or statistical comparisons |
| GET | `/recommendations/models`, `/recommendations/comparison` | Supported modes and model metrics |
| GET | `/recommendations/products` | Product search by `model` and `q`, with `offset` and `limit` |
| GET | `/recommendations/products/{product_id}`, `/recommendations/customers/{customer_id}` | Ranked products for a `model` and `top_n` (1–50) |
| GET | `/forecasting/categories`, `/forecasting/products?category=...` | Categories and their eligible products |
| GET | `/forecasting/products/{product_id}`, `/forecasting/products/{product_id}/history` | Future total or test-period results |
| GET | `/forecasting/products/{product_id}/stores` | Estimated units by store and the historical sales shares used to allocate them |
| GET | `/forecasting/evaluation` | Overall test scores for the three methods |

## Testing and maintenance

```powershell
uv run --no-sync python -m pytest -q -rs
uv run --no-sync ruff check api streamlit_app tests src/pipelines/export_churn_test_features.py src/pipelines/export_recommendation_seeds.py
docker compose config --quiet
```

Tests cover real model loading, the saved XGBoost forecast, recommendation scores and purchase exclusions, unknown IDs, missing files, churn input checks, historical chart data, and the Random Fill form. Artifact tests need the files listed above. Read the skip reasons when running on a partial copy.

After changing dependencies, update the lock and export the two runtime groups again.

```powershell
uv lock
uv export --frozen --no-default-groups --group api --group dashboard --no-emit-project --output-file requirements.txt
```

## Limitations and next steps

All five tabs use one historical data release. Customer coverage changes by module and time window. The demand model estimates 30-day product totals across stores. The store view assumes recent sales shares will continue. Campaign results describe associations. A later release could add a new time period, check whether the churn model still works well, and measure forecast error when new actual sales arrive. Restart the services after replacing artifact files so that both caches use the same release.
