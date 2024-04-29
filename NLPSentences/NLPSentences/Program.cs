using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MathNet.Numerics.Statistics;
using Microsoft.ML.Transforms;
using NLPSentences.Common;

// Initialize the MLContext
var context = new MLContext
{
    // (Optional) Use GPU
    GpuDeviceId = 0,
    FallbackToCpu = false
};

// Log training output
context.Log += (o, e) => {
    if (e.Source.Contains("NasBertTrainer"))
        Console.WriteLine(e.Message);
};

// Load the data into an IDataView
var columns = new[]
{
    new TextLoader.Column("search_term",DataKind.String,3),
    new TextLoader.Column("relevance",DataKind.Single,4),
    new TextLoader.Column("product_description",DataKind.String,5)
};

var loaderOptions = new TextLoader.Options()
{
    Columns = columns,
    HasHeader = true,
    Separators = new[] { ',' },
    MaxRows = 1000 // Dataset has 75k rows. Only load 5k for quicker training
};

var dataPath = @"../../../data/home-depot-sentence-similarity.csv";
var textLoader = context.Data.CreateTextLoader(loaderOptions);
var data = textLoader.Load(dataPath);

// Split data into 80% training, 20% testing
var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Print a description of the data to console
ConsoleHelper.ShowDataViewInConsole(context, dataSplit.TrainSet);

// Define the processing pipeline with relevance column as our label
var pipeline =
    context.Transforms.ReplaceMissingValues("relevance", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
    .Append(context.Regression.Trainers.SentenceSimilarity(labelColumnName: "relevance", sentence1ColumnName: "search_term", sentence2ColumnName: "product_description"));

// Train the model
var model = pipeline.Fit(dataSplit.TrainSet);

// Use the model to make predictions on the test dataset
var predictions = model.Transform(dataSplit.TestSet);

// Evaluate the model
Evaluate(predictions, "relevance", "Score");

// Save the model so we don't have to keep retraining
context.Model.Save(model, data.Schema, "../../../model.zip");

void Evaluate(IDataView predictions, string actualColumnName, string predictedColumnName)
{
    var actual =
        predictions.GetColumn<float>(actualColumnName)
            .Select(x => (double)x);
    var predicted =
        predictions.GetColumn<float>(predictedColumnName)
            .Select(x => (double)x);
    var corr = Correlation.Pearson(actual, predicted);
    Console.WriteLine($"Pearson Correlation: {corr}");
}