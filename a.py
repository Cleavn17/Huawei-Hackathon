from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_solution, load_problem_data

plt.style.use('dark_background')

def analyze_json(json_data):
    df = pd.DataFrame(json_data)

    action_counter = Counter(df['action'])
    server_type_counter = Counter(df['server_generation'].apply(lambda x: x.split('.')[0]))
    server_generation_counter = Counter(df['server_generation'])
    datacenter_counter = Counter(df['datacenter_id'])

    actions_per_timestep = df.groupby(['time_step', 'action']).size().unstack(fill_value=0)
    upgrades_per_timestep = df[df['action'] == 'buy'].groupby(['time_step', 'server_generation']).size().unstack(
        fill_value=0)

    analysis = {
        'Overall Counts': {
            'Actions': dict(action_counter),
            'Server Types': dict(server_type_counter),
            'Server Generations': dict(server_generation_counter),
            'Datacenters': dict(datacenter_counter)
        },
        'Time-based Analysis': {
            'Actions per Timestep': actions_per_timestep,
            'Upgrades per Timestep': upgrades_per_timestep
        },
        'Raw Data': df
    }

    return analysis


def visualize_results(analysis):
    fig, axs = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle('Server Fleet Management Analysis', fontsize=24)

    # Overall counts
    sns.barplot(x=list(analysis['Overall Counts']['Actions'].keys()),
                y=list(analysis['Overall Counts']['Actions'].values()), ax=axs[0, 0])
    axs[0, 0].set_title('Actions Distribution', fontsize=16)
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=list(analysis['Overall Counts']['Server Generations'].keys()),
                y=list(analysis['Overall Counts']['Server Generations'].values()), ax=axs[1, 0])
    axs[1, 0].set_title('Server Generations Distribution', fontsize=16)
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Time-based analysis
    actions_df = analysis['Time-based Analysis']['Actions per Timestep']
    actions_df.plot(kind='area', stacked=True, ax=axs[0, 1])
    axs[0, 1].set_title('Actions per Timestep', fontsize=16)
    axs[0, 1].set_xlabel('Timestep')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].legend(title='Action', loc='center left', bbox_to_anchor=(1, 0.5))

    upgrades_df = analysis['Time-based Analysis']['Upgrades per Timestep']
    upgrades_df.plot(kind='area', stacked=True, ax=axs[1, 1])
    axs[1, 1].set_title('Upgrades (Buys) per Timestep', fontsize=16)
    axs[1, 1].set_xlabel('Timestep')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend(title='Server Generation', loc='center left', bbox_to_anchor=(1, 0.5))

    # Additional insights
    df = analysis['Raw Data']
    sns.countplot(data=df, x='datacenter_id', hue='server_generation', ax=axs[0, 2])
    axs[0, 2].set_title('Server Generations by Datacenter', fontsize=16)
    axs[0, 2].set_xlabel('Datacenter')
    axs[0, 2].set_ylabel('Count')
    axs[0, 2].legend(title='Server Generation', loc='center left', bbox_to_anchor=(1, 0.5))

    action_proportions = df['action'].value_counts(normalize=True)
    axs[1, 2].pie(action_proportions.values, labels=action_proportions.index, autopct='%1.1f%%')
    axs[1, 2].set_title('Proportion of Actions', fontsize=16)

    fig.savefig("plot-sb.png")

    plt.tight_layout()
    plt.show()


def main():
    path = 'neo/1097.json'
    solution, prices = load_solution(path)

    result = analyze_json(solution)

    for category, subcategories in result['Overall Counts'].items():
        print(f"\n{category}:")
        for item, count in subcategories.items():
            print(f"  {item}: {count}")

    print("\nTime-based Analysis:")
    print(result['Time-based Analysis']['Actions per Timestep'])
    print("\nUpgrades per Timestep:")
    print(result['Time-based Analysis']['Upgrades per Timestep'])

    visualize_results(result)

    # Analysis of problem data
    demand, datacenters, servers, selling_prices = load_problem_data()
    print("\nDatacenter Information:")
    print(datacenters)
    print("\nServer Generations:")
    print(servers['server_generation'].unique())


if __name__ == "__main__":
    main()
