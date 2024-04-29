
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using StockMarketPrediction.Common;
using StockMarketPrediction.DataStructures;
using XPlot.Plotly;


namespace StockMarketPrediction
{
    internal static class Program
    {
        private static readonly string LabelColumnName = "AdjClose";
        private static readonly uint ExperimentTime = 60;

        static void Main(string[] args)
        {
            
            var stockPath = "../../../Data/msft-train-data.csv";

            // Fetch the data if we do not have it already. You can change these parameters to get different periods
            if (!File.Exists(stockPath))
            {
                var contents =  new HttpClient()
                    .GetStringAsync("https://query1.finance.yahoo.com/v7/finance/download/MSFT?period1=1514764800&period2=1627776000&interval=1d&events=history&includeAdjustedClose=true");

                File.WriteAllText(stockPath, contents.GetAwaiter().GetResult());
            }

            var context = new MLContext();
            var trainingDataView = context.Data.LoadFromTextFile<StockPrice>( "../../../Data/msft-train-data.csv", hasHeader: true, separatorChar: ',', );
            
            // Display the data in a plot
            var chart = Chart.Plot(
                new Scatter()
                {
                    x = trainingDataView.GetColumn<DateTime>("Date"),
                    y = trainingDataView.GetColumn<float>("AdjClose")
                }
            );
            
            chart.Show();
            
            ConsoleHelper.ShowDataViewInConsole(context, trainingDataView);
            
            var model = context.Forecasting.ForecastBySsa(
                outputColumnName: "Forecast",   // Name for our prediction  column
                inputColumnName: "AdjClose",    // This is our label column
                windowSize: 100,                // Look at the previous 100 values
                seriesLength: 365,              // Amount of data points to keep in buffer
                trainSize: 800,                 // Take the first 800 values in the data
                horizon: 3                      // Generate three predictions
            );
            
            var transformer = model.Fit(trainingDataView);
            
            var forecastEngine = transformer.CreateTimeSeriesEngine<Data, ForecastResult>(context);

            var forecast = forecastEngine.Predict();
            
            // Output the three predictions
            Console.WriteLine($"Forecast values : {forecast.Forecast[0]} {forecast.Forecast[1]} {forecast.Forecast[2]}");
            
            // Now specify confidence boundaries
            model = context.Forecasting.ForecastBySsa(
                outputColumnName: "Forecast",   // Name for our prediction  column
                inputColumnName: "AdjClose",    // This is our label column
                windowSize: 100,                // Look at the previous 100 values
                seriesLength: 365,              // Amount of data points to keep in buffer
                trainSize: 800,                 // Take the first 800 values in the data
                horizon: 3,                     // Generate three predictions
                confidenceLevel: 0.95f,         // Confidence level we want for the forecast 
                confidenceLowerBoundColumn: "LowerBound",   //  Name of the confidence interval lower bound column
                confidenceUpperBoundColumn: "UpperBound"    //  name of the confidence interval upper bound column
            );
            
            // Fit the data model again as it has changed
            transformer = model.Fit(trainingDataView); 
            
            var boundedForecastEngine = transformer.CreateTimeSeriesEngine<Data, BoundedForecastResult>(context);

            var boundedForecast = boundedForecastEngine.Predict();

            // Output the results to the console
            Console.WriteLine();
            Console.WriteLine("Bounded Forecasts");
            foreach (var idx in Enumerable.Range(0, 3))
            {
                Console.WriteLine($"Result {idx}: " +
                                  $"Forecast: {boundedForecast.Forecast[idx]} " +
                                  $"\t LBound: {boundedForecast.LowerBound[idx]}" +
                                  $"\t UBound: {boundedForecast.UpperBound[idx]}");
            } 
            
        }
    }
}