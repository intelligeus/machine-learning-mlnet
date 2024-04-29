

using Microsoft.Extensions.Hosting;

// Launch our server/cluster with distributed transactions enabled
await Host.CreateDefaultBuilder(args)
    .UseOrleans(siloBuilder =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .AddMemoryGrainStorageAsDefault()
            .UseTransactions();
    })
    .RunConsoleAsync();