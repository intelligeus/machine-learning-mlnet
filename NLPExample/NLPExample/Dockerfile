FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["NLPExample/NLPExample.csproj", "NLPExample/"]
RUN dotnet restore "NLPExample/NLPExample.csproj"
COPY . .
WORKDIR "/src/NLPExample"
RUN dotnet build "NLPExample.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "NLPExample.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "NLPExample.dll"]
