using Microsoft.ML.Data;

namespace LinearRegression.DataStructures
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}