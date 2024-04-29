FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["BankServer/BankServer.csproj", "BankServer/"]
RUN dotnet restore "BankServer/BankServer.csproj"
COPY . .
WORKDIR "/src/BankServer"
RUN dotnet build "BankServer.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "BankServer.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "BankServer.dll"]
