using AccountGrains.interfaces;
using Orleans.Concurrency;
using Orleans.Transactions.Abstractions;

namespace AccountGrains;

[GenerateSerializer]
public record AccountBalance
{
    [Id(0)]
    public decimal CurrentBalance { get; set; } = 1000;
}

[Reentrant]
public class AccountGrain : Grain, IAccountGrain
{
    
    private readonly ITransactionalState<AccountBalance> _currentBalance;
    
    public AccountGrain(
        [TransactionalState("CurrentBalance")] ITransactionalState<AccountBalance> balance) =>
        _currentBalance = balance ?? throw new ArgumentNullException(nameof(balance));
    
    // Withdraw money from the account. We have done a check prior to this call to 
    // make sure the account current balance is sufficient
    public Task Withdraw(decimal amount) =>
        _currentBalance.PerformUpdate(acc =>
        {
            acc.CurrentBalance -= amount;
        });
    

    public Task Deposit(decimal amount) =>
        _currentBalance.PerformUpdate(account  => account.CurrentBalance += amount);


    public Task<decimal> BankDeposit(decimal amount) =>
        _currentBalance.PerformUpdate(account => account.CurrentBalance += amount);


    public Task<decimal> CheckBalance() => _currentBalance.PerformRead(acc => acc.CurrentBalance);
        


}