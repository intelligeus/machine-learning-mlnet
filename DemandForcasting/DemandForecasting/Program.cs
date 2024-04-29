using DemandForecasting.Common;
using DemandForecasting.DataStructures;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace DemandForecasting
{

    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"Data";
        private static readonly string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/hour.csv";

        private static readonly string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        // private static readonly string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../Models";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/RentalDemand.zip";
        // private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static readonly string LabelColumnName = "TotalRentals";
        private static readonly uint ExperimentTime = 60;


        static void Main(string[] args)
        {

            var context = new MLContext();

            BuildTrainEvaluateAndSaveModel(context);



            var forecastingPipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedRentals",
                inputColumnName: "TotalRentals",
                windowSize: 7,
                seriesLength: 30,
                trainSize: 365,
                horizon: 7,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundRentals",
                confidenceUpperBoundColumn: "UpperBoundRentals");
        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext context)
        {
            
            var trainingDataView = context.Data.LoadFromTextFile<TrialRental>(TrainDataPath, hasHeader: true, separatorChar: ',');

            ConsoleHelper.ShowDataViewInConsole(context, trainingDataView);
            
            // STEP 3: Initialize our user-defined progress handler that AutoML will 
            // invoke after each model it produces and evaluates.
            var progressHandler = new RegressionExperimentProgressHandler();
            
            // STEP 4: Run AutoML regression experiment
            ConsoleHelper.ConsoleWriteHeader("=============== Training the model ===============");
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");
            ExperimentResult<RegressionMetrics> experimentResult = context.Auto()
                .CreateRegressionExperiment(ExperimentTime)
                .Execute(trainingDataView, LabelColumnName, progressHandler: progressHandler);

            // Print top models found by AutoML
            Console.WriteLine();
            PrintTopModels(experimentResult);
            
        }


        private static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            var assemblyFolderPath = dataRoot.Directory.FullName;

            return Path.Combine(assemblyFolderPath, relativePath);

        }
        
        /// <summary>
        /// Print top models from AutoML experiment.
        /// </summary>
        private static void PrintTopModels(ExperimentResult<RegressionMetrics> experimentResult)
        {
            // Get top few runs ranked by R-Squared.
            // R-Squared is a metric to maximize, so OrderByDescending() is correct.
            // For RMSE and other regression metrics, OrderByAscending() is correct.
            var topRuns = experimentResult.RunDetails
                .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.RSquared))
                .OrderByDescending(r => r.ValidationMetrics.RSquared).Take(3);

            Console.WriteLine("Top models ranked by R-Squared --");
            ConsoleHelper.PrintRegressionMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }
        }
    }
}


