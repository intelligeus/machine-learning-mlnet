using Microsoft.ML.Data;
using MultiImageClassification.Tensorflow;

namespace MultiImageClassification.Model
{
    public class ImagePrediction
    {
        [ColumnName(TensorflowScorer.InceptionSettings.OutputTensorName)]
        public float[] PredictedLabels;
    }
}
