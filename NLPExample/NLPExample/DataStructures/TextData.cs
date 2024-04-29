using Microsoft.ML.Data;

namespace NLPExample.DataStructures;

public class TextData
{
    [LoadColumn(2)]
    public string ProductTitle;
    [LoadColumn(3)]
    public string SearchTerm;
    [LoadColumn(4)]
    public float Relevance;
}