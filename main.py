import ai_benchmark
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process command line arguments')
    parser.add_argument('--run', help='Specify a run type, eg: --run=inference,training,micro,all')
    parser.add_argument('--model', help='Specify a model, eg: --model=MobileNet-V2,Inception-V3,...')
    args = parser.parse_args()

    model = args.model
    run_model = args.run
    print(f'model:{model}')
    print(f'run_model:{run_model}')
    benchmark = ai_benchmark.AIBenchmark()
    if run_model == "inference":
        results = benchmark.run_inference(precision="high", model=model)
    elif run_model == "training":
        results = benchmark.run_training(precision="high", model=model)
    elif run_model == "micro":
        results = benchmark.run_micro(precision="high", model=model)
    elif run_model == "all":
        results = benchmark.run(precision="high", model=model)
    else:
        results = benchmark.run_training(precision="high", model=model)
    print(results)

if __name__ == "__main__":
    # execute only if run as a script
    main()
