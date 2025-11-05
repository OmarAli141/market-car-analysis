import pathlib 
import polars as pl

def prepare_car_reviews_data(path: pathlib.Path, vehicle_years: list[int] = [2017]):
    """
    Prepare the car reviews dataset for ChromaDB
    """

    # Define the schema to ensure proper data types are enforced
    dtypes = {
        "": pl.Int64,
        "Review_Date": pl.Utf8,
        "Author_Name": pl.Utf8,
        "Vehicle_Title": pl.Utf8,
        "Review_Title": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64
    }

    # Scan the car reviews dataset and ensure the data types are correct
    car_reviews = pl.scan_csv("D:\\market_car_analysis\\data\\archive", dtypes=dtypes)

    # Extract the vehicle title and year as new columns
    # Filter on selected years
    car_review_db_data = (
        car_reviews.with_columns(
            [
                (
                    pl.col("Vehicle_Title").str.split(
                        by=" ").list.get(0).cast(pl.Int64)
                ).alias("Vehicle_Year"),
                (pl.col("Vehicle_Title").str.split(by=" ").list.get(1)).alias(
                    "Vehicle_Model"
                ),
            ]
        )
        .filter(pl.col("Vehicle_Year").is_in(vehicle_years))
        .select(["Review_Title", "Review", "Rating", "Vehicle_Year", "Vehicle_Model"])
        .sort(["Vehicle_Model", "Rating"])
        .collect()
    )

    ids = [f"review{i}" for i in range(car_review_db_data.shape[0])]
    documents = car_review_db_data["Review"].to_list()
    metadatas = car_review_db_data.drop("Review").to_dicts()

    return {"ids": ids, "documents": documents, "metadatas": metadatas}
