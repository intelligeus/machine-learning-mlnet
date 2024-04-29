using Microsoft.ML.Data;

namespace BinaryClassification.DataStructures;

public class RelevantInfo
{
    [LoadColumn(2, 31), ColumnName("Features")]
    public float[] FeaturesVector { get; set; }

    [LoadColumn(1)]
    public bool Status { get; set; }
    
}