using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace CodeProject.AI.Modules.DotNetSimple
{
    internal static class Program
    {
        static async Task Main(string[]? args)
        {
            DotNetSimpleWorker.ProcessArguments(args);

            IHost host = Host.CreateDefaultBuilder(args)
                             .ConfigureServices(services =>
                              {
                                  services.AddHostedService<DotNetSimpleWorker>();
                              })
                              .Build();

            #pragma warning disable CA2007 // Consider calling ConfigureAwait on the awaited task
            await host.RunAsync();
            #pragma warning restore CA2007 // Consider calling ConfigureAwait on the awaited task
        }
    }
}