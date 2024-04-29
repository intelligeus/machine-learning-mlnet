using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MathNet.Numerics.Statistics;
using Microsoft.ML.Transforms;
using NLPExample.Common;
using NLPExample.DataStructures;

// Initialize MLContext
var context = new MLContext
{
    // Use GPU but allow Cpu if none is detected
    GpuDeviceId = 0, 
    FallbackToCpu = true
};

// Log training output
context.Log += (o, e) => {
    if (e.Source.Contains("NasBertTrainer"))
        Console.WriteLine(e.Message);
};

// Load the data into an IDataView. We are only taking 3k rows for this example 
// to speed up the training process 
var trainingDataView = context.Data.LoadFromTextFile<TextData>( "home-depot-train.csv",
    new TextLoader.Options(){HasHeader = true, MaxRows = 3000, Separators = new []{','}});


// Print a description of the data to console
ConsoleHelper.ShowDataViewInConsole(context, trainingDataView);

// Split the 3k rows to an 80:20 training:test split
var trainingData = context.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

// Define the pipeline we are going to use
var pipeline =
    context.Transforms.ReplaceMissingValues("Relevance",
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
        .Append(context.Regression.Trainers.SentenceSimilarity(labelColumnName: "Relevance",
            sentence1ColumnName: "SearchTerm", 
            sentence2ColumnName: "ProductTitle"));


// Train the model
var model = pipeline.Fit(trainingData.TrainSet);

// Use the model to make predictions on the test dataset
var predictions = model.Transform(trainingData.TestSet);

// Evaluate the model
Evaluate(predictions, "Relevance", "Score");

// Save the model
context.Model.Save(model, trainingDataView.Schema, "model.zip");

void Evaluate(IDataView dataView, string actualColumnName, string predictedColumnName)
{
    var actual =
        dataView.GetColumn<float>(actualColumnName)
            .Select(x => (double)x);
    var predicted =
        dataView.GetColumn<float>(predictedColumnName)
            .Select(x => (double)x);
    var corr = Correlation.Pearson(actual, predicted);
    
    Console.WriteLine($"Pearson Correlation: {corr}");
}