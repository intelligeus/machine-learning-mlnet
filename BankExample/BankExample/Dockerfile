FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["BankExample/BankExample.csproj", "BankExample/"]
RUN dotnet restore "BankExample/BankExample.csproj"
COPY . .
WORKDIR "/src/BankExample"
RUN dotnet build "BankExample.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "BankExample.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "BankExample.dll"]
