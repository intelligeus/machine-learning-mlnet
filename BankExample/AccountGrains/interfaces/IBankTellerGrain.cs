namespace AccountGrains.interfaces;

public interface IBankTellerGrain : IGrainWithIntegerKey
{
    //Task<decimal> DepositFunds(string accountId, decimal amount);
    Task<decimal> DepositFunds(string accountId, decimal amount);
}