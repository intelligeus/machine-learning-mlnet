
using AccountGrains.interfaces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

using var host = Host.CreateDefaultBuilder(args)
    .UseOrleansClient(client =>
    {
        client.UseLocalhostClustering()
            .UseTransactions();
    })
    .UseConsoleLifetime()
    .Build();

await host.StartAsync();

var client = host.Services.GetRequiredService<IClusterClient>();
var transactionClient = host.Services.GetRequiredService<ITransactionClient>();
var lifetime = host.Services.GetRequiredService<IHostApplicationLifetime>();

// Set of random account names to use
var random = Random.Shared;
var accounts = new[]
{
    "Aniyah Pacheco",
    "Erik Bridges",
    "Elora Harrington",
    "Omari Bernard",
    "Barbara Ware",
    "Tadeo Vega",
    "Dakota Reese",
    "Alijah Hernandez",
    "Camila Garrett",
    "Kairo Kim",
    "Gabriella Chung",
    "Ira McKay",
    "Leanna Sanchez",
    "Joseph Randall",
    "Christina Carter",
    "Maverick Dickerson",
    "Opal Peck",
    "Yousef Hudson",
    "Kamila Duncan",
    "Avery Ayers",
    "Simone Rangel",
    "Saint Johns"
};

while (!lifetime.ApplicationStopping.IsCancellationRequested)
{

    var fromIndex = random.Next(accounts.Length);
    var toIndex = random.Next(accounts.Length);
    // It's possible we get the same account for withdrawal and deposit so if we do we need to change it to
    // another account
    if (toIndex == fromIndex)
    {
        toIndex = (toIndex + 1) % accounts.Length;
    }
    
    /*
     * Our grains have string ids so we use the names as the grain primary key (the id we use to fetch it)
     */
    var fromKey = accounts[fromIndex];
    var toKey = accounts[toIndex];
    
    var fromAccount = client.GetGrain<IAccountGrain>(fromKey);
    var toAccount = client.GetGrain<IAccountGrain>(toKey);
    
    try
    {
        var transferAmount = random.Next(500);

        var currentBalance = await fromAccount.CheckBalance();
        
        // Check if the account has sufficient funds to cover the withdrawal if not then we go to the 
        // Bank and give the teller 1000 Calamari Flans to deposit
        if (currentBalance < transferAmount)
        {
            Console.WriteLine($"{fromAccount.GetPrimaryKeyString()} Withdrawal request for {transferAmount} Calamari Flan will fail as Current Balance is {currentBalance}");
            var teller = client.GetGrain<IBankTellerGrain>(0);
            decimal bal = 0;
            // We are doing a write so we need a tx here 
            await transactionClient.RunTransaction(
                TransactionOption.Create,
                async () =>
                {
                    bal = await teller.DepositFunds(fromKey, 1000);
                    
                });
            // Log the updated balance
            Console.WriteLine($"{fromKey} After BANK deposit Current Balance is {bal}");
        }
        
        // Now we can make the actual withdrawal and deposit knowing the from account will not go negative
        await transactionClient.RunTransaction(
            TransactionOption.Create,
            async () =>
            {
                await fromAccount.Withdraw(transferAmount);
                await toAccount.Deposit(transferAmount);
            });

        // Log the new balances to the console
        var fromBalance = await fromAccount.CheckBalance();
        var toBalance = await toAccount.CheckBalance();
        Console.WriteLine(
            $"Transferred {transferAmount} Calamari Flan from {fromKey} to " +
            $"{toKey}.\n{fromKey} current balance: {fromBalance}\n{toKey} current balance: {toBalance}\n");
    }
    catch (Exception exception)
    {
        Console.WriteLine(
            $"Error transferring Calamari Flan from " +
            $"{fromKey} to {toKey}: {exception.Message}");

        if (exception.InnerException is { } inner)
        {
            Console.WriteLine($"\tInnerException: {inner.Message}\n");
        }

        Console.WriteLine();
    }

    // Sleep and run again
    await Task.Delay(TimeSpan.FromMilliseconds(250));
    
}