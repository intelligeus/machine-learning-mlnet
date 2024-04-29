// Load the data into an IDataView. We are only taking 3k rows for this example 
// to speed up the training process 

using BinaryClassification.Common;
using BinaryClassification.DataStructures;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using NLPExample.DataStructures;

uint ExperimentTime = 60;

var context = new MLContext();

context.Log += (_, e) => {
    if (e.Source.Equals("AutoMLExperiment"))
    {
        Console.WriteLine(e.RawMessage);
    }
};


// var trainData = context.Data.LoadFromTextFile<RelevantInfo>("../../../Data/data 2.csv", hasHeader: true, separatorChar: ',');
//
// var pipe = context.Transforms
//     .NormalizeMinMax("Features")
//     .AppendCacheCheckpoint(context)
//     .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Status", featureColumnName: "Features"));
//
//
// // var targetMap = new Dictionary<string, bool> { { "+", true }, { "-", false } };
// //
// // var pipe = context.Transforms.NormalizeMinMax("Features")
// //     .AppendCacheCheckpoint(context)
// //     .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Status", featureColumnName: "Features"));
// //
// //
//
// var model = pipe.Fit(trainData);

ColumnInferenceResults columnInference =
    context.Auto().InferColumns("../../../Data/cc_approvals.csv", labelColumnName: "ApprovalStatus", groupColumns: true);

// Create text loader
TextLoader loader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);

// Load data into IDataView
IDataView trainingDataView = loader.Load("../../../Data/cc_approvals.csv");

// var trainingDataView = context.Data.LoadFromTextFile<TextData>( "../../../Data/cc_approvals.csv",
//     new TextLoader.Options(){HasHeader = true, MaxRows = 3000, Separators = new []{','}});


IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(context);

// Train Model
ITransformer mlModel = TrainModel(context, trainingDataView, trainingPipeline);

// Evaluate quality of Model
Evaluate(context, trainingDataView, trainingPipeline);


var progressHandler = new BinaryExperimentProgressHandler();
            
// STEP 4: Run AutoML regression experiment
ConsoleHelper.ConsoleWriteHeader("=============== Training the model ===============");
Console.WriteLine($"Running AutoML Binary Classification experiment for {ExperimentTime} seconds...");
ExperimentResult<BinaryClassificationMetrics> experimentResult = context.Auto()
    .CreateBinaryClassificationExperiment(ExperimentTime)
    .Execute(trainingDataView, "ApprovalStatus", progressHandler: progressHandler);

// Print top models found by AutoML
Console.WriteLine();



//var preview = trainingDataView.Preview();
// Print a description of the data to console
//
// Split the 3k rows to an 80:20 training:test split
//var trainingData = context.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

SweepablePipeline pipeline =
    context.Auto().Featurizer(trainingDataView, columnInformation: columnInference.ColumnInformation)
        .Append(context.Auto().Regression(labelColumnName: columnInference.ColumnInformation.LabelColumnName));
        
AutoMLExperiment experiment = context.Auto().CreateExperiment().SETP;


experiment
    .SetPipeline(pipeline)
    .SetRegressionMetric(RegressionMetric.RSquared, labelColumn: columnInference.ColumnInformation.LabelColumnName)
    .SetTrainingTimeInSeconds(60)
    .SetDataset(trainingDataView);
    
TrialResult experimentResults = await experiment.RunAsync();

Console.WriteLine("pppp");
    
IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Survived", "Survived")
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "col0", "PassengerId", "Sex", "Age", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Family_size", "Title_1", "Title_2", "Title_3", "Title_4", "Emb_1", "Emb_2", "Emb_3" }))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);
            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Survived", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Survived")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
{
    Console.WriteLine("=============== Training  model ===============");

    ITransformer model = trainingPipeline.Fit(trainingDataView);

    Console.WriteLine("=============== End of training process ===============");
    return model;
}
void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
{
    // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
    // in order to evaluate and get the model's accuracy metrics
    Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
    var crossValidationResults = mlContext.BinaryClassification.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Survived");
    PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
}