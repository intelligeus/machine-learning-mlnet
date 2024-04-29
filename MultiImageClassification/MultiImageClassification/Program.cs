// See https://aka.ms/new-console-template for more information

using MultiImageClassification.Common;
using MultiImageClassification.Tensorflow;

var  dataRoot = new FileInfo(typeof(Program).Assembly.Location);
var assemblyFolderPath = dataRoot.Directory?.FullName;

// Set up the paths
var assetsRelativePath = @"../../../assets";
var assetsPath = Path.Combine(assemblyFolderPath, assetsRelativePath);

// This file contains the name of the images to test and it's label
var tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
// Location of the images we are going to make predictions against
var imagesFolder = Path.Combine(assetsPath, "inputs", "images");
// Location of the Inception 5H model
var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
// Related inception data
var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");


try
{
    var modelScorer = new TensorflowScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);
    modelScorer.Score();

}
catch (Exception ex)
{
    ConsoleHelper.ConsoleWriteException(ex.ToString());
}

ConsoleHelper.ConsolePressAnyKey();


