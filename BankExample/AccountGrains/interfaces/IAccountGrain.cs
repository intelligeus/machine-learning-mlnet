namespace AccountGrains.interfaces;


public interface IAccountGrain : IGrainWithStringKey
{
    
    [Transaction(TransactionOption.Join)]
    Task Withdraw(decimal amount);

    [Transaction(TransactionOption.Join)]
    Task Deposit(decimal amount);
    
    [Transaction(TransactionOption.CreateOrJoin)]
    Task<decimal> BankDeposit(decimal amount);

    [Transaction(TransactionOption.CreateOrJoin)]
    Task<decimal> CheckBalance();
}