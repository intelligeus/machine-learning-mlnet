using AccountGrains.interfaces;

namespace AccountGrains;

public class BankTellerGrain : Grain, IBankTellerGrain
{
    public Task<decimal> DepositFunds(string accountId, decimal amount){

        // We get the PK for the account grain so we look it up
        var customer = GrainFactory.GetGrain<IAccountGrain>(accountId);

        // Update the current balance on the account using the current transaction scope
        customer.BankDeposit(1000);
        
        // Return the current balance for the account
        return customer.CheckBalance();
    }

}