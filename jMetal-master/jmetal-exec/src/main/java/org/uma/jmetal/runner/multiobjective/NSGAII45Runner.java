package org.uma.jmetal.runner.multiobjective;

import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.codehaus.jackson.JsonGenerationException;
import org.codehaus.jackson.map.JsonMappingException;
import org.codehaus.jackson.map.ObjectMapper;
import org.json.simple.JSONObject;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAII45;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.retorno;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.user;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.*;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.qualityindicator.impl.*;
import org.uma.jmetal.problem.multiobjective.dtlz.*;
import org.uma.jmetal.problem.multiobjective.wfg.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import java.lang.Math;

/**
 * Class to configure and run the implementation of the NSGA-II algorithm
 * included in {@link NSGAII45}
 *
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
public class NSGAII45Runner extends AbstractAlgorithmRunner {
	/**
	 * @param args Command line arguments.
	 * @throws JMetalException
	 * @throws IOException
	 */
	public static void main(String[] args) throws JMetalException, IOException {

		long tNow = System.currentTimeMillis();
		int numObj = 3;
		int numExecution = 1;

		String surrogateMethods[] = { "Batch", "Online" };
		String problemTags[] = { "DTLZ", "WFG" };
		// Limits number of version of problems to be simulated.
		int startVersionProblems = 1;
		int maxVersionsProblems = 3;

//		case 0: "NO-SURROGATE";
//		case 1: "SVM";
//		case 2: "RAMDOMFOREST";
//		case 3: "TREE";
//		case 4: "LSTM-FIXED";
//		case 5: "RNN-FIXED";
//		case 6: "RANDOM";
		
		int nodesHiddenLayerNN = 92;
		int numOfEpochs = 10;
		boolean avrOptimizationNN = false;
		// If Neural Network, more information is integrated in surrogateName
		String[] surrogateNNs = { "LSTM-FIXED" , "RNN-FIXED" };
		String suffixNN = "";
		suffixNN += 	"_amntNodesHidden=" + Integer.toString(nodesHiddenLayerNN) +
								"_amntEpochs=" + Integer.toString(numOfEpochs);

		ArrayList<String> surrogateNames = new ArrayList<String>();
		for(String surNN : surrogateNNs) {
			for(int avrOpt = 0; avrOpt <= 1; avrOpt++) {
				for(int timestep = 22; timestep <= 22; timestep += 1) {
					String now = surNN + suffixNN + "_ts=" + Integer.toString(timestep);
					if(avrOpt == 1)
						now += "_avr";
					surrogateNames.add(now);
					System.out.println(now);
				}
			}
		}
		
		for(String surrogateName : surrogateNames)
			for(String surrogateMethod : surrogateMethods)
				for(String problemTag : problemTags) {
					int amntOfVersions = Math.min(7, maxVersionsProblems);
					if (problemTag == "WFG")
						amntOfVersions = Math.min(9, maxVersionsProblems);
					for(int problemId = startVersionProblems; problemId <= amntOfVersions; problemId++)
						executeExperiment(numObj, (int)tNow, numExecution, args,
								surrogateName, surrogateMethod,
								problemTag, problemTag + Integer.toString(problemId));
				}
		
	}
	
	public static void executeExperiment(int numObj, int tNow, int numExecution, String[] args,
			String surrogateName, String surrogateMethod,
			String problemTag, String problemName) throws JMetalException, IOException {
		Problem<DoubleSolution> problem;
		Algorithm<List<DoubleSolution>> algorithm;
		CrossoverOperator<DoubleSolution> crossover;
		MutationOperator<DoubleSolution> mutation;
		SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;
		String referenceParetoFront = "";
		InvertedGenerationalDistance indice;
		ArrayList igds = new ArrayList<>();
		String algoritmo = null;

		int maxEval = 10000;
		int populationSize = -1;
		String surrogate = "Surrogate_" + surrogateName + "_" + surrogateMethod;
		boolean online = false;

		if (surrogateMethod.equals("Online"))
			online = true;

		// Identifier of the current simulation!
		String execIdentifier = "out_NSGAII45_IGD_" + problemTag + "_" + surrogate + "_" + "Obj-"
				+ Integer.toString(numObj) + "_" + "EvalPopulation-" + Integer.toString(maxEval) + "_"
				+ "PopulationSize-" + Integer.toString(populationSize) + "_" + "timeStamp-"
				+ String.valueOf(tNow);
		System.out.println("----------| " + execIdentifier + " |----------");

		for (int i = 0; i < numExecution; i++) {
			String problemNameOld;
			if (args.length == 1) {
				problemNameOld = args[0];
			} else if (args.length == 2) {
				problemNameOld = args[0];
				referenceParetoFront = args[1];
			} else {
				if (problemName.startsWith("DTLZ"))
					problemNameOld = "org.uma.jmetal.problem.multiobjective.dtlz." + problemName;
				else
					problemNameOld = "org.uma.jmetal.problem.multiobjective.wfg." + problemName;
				referenceParetoFront = "";
			}

			// problem = ProblemUtils.<DoubleSolution> loadProblem(problemName);
			// problem = problem.createSolution();
			problem = getProblem(problemName, numObj);

			referenceParetoFront = "../jmetal-problem/src/test/resources/pareto_fronts/" + problemName
					+ "." + Integer.toString(numObj) + "D.pf";
			// referenceParetoFront =
			// "/home/joe/MESTRADO_LINUX/eclipse-workspace/jMetal-master.zip_expanded/jMetal-master/jmetal-problem/src/test/resources/pareto_fronts/DTLZ2.10D.pf";
			if (numObj == 10)
				populationSize = 764;
			else if (numObj == 3)
				populationSize = 92;

			ArrayList array = new ArrayList<>(1);
			array.add(numObj);

			user userObject = new user(
					surrogateName + "_" + problemName + "_" + surrogateMethod,
					problemTag,
					new ArrayList<>(),
					array
			);
			ArrayList SwarmInicio = http("http://127.0.0.1:5000/classificador", userObject);

			double crossoverProbability = 0.9;
			double crossoverDistributionIndex = 20.0;
			crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex);

			double mutationProbability = 1.0 / problem.getNumberOfVariables();
			double mutationDistributionIndex = 20.0;
			mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex);

			selection = new BinaryTournamentSelection<DoubleSolution>(
					new RankingAndCrowdingDistanceComparator<DoubleSolution>());

			algorithm = new NSGAII45<DoubleSolution>(problem, maxEval, populationSize, crossover,
					mutation, selection, new SequentialSolutionListEvaluator<DoubleSolution>(),
					online, surrogateName == "NO-SURROGATE");

			AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm).execute();

			List<DoubleSolution> population = algorithm.getResult();
			// long computingTime = algorithmRunner.getComputingTime() ;

			/*
			 * new SolutionListOutput(population) .setSeparator("\t")
			 * .setVarFileOutputContext(new DefaultFileOutputContext(surrogate
			 * +"VAR"+Integer.toString(maxEval)+"Eval"+nameProblem+"_"+Integer.toString(
			 * numObj)+"OBJ_Population_"+Integer.toString(populationSize)+"_EXC"+i+".tsv"))
			 * .setFunFileOutputContext(new DefaultFileOutputContext(surrogate
			 * +"FUN"+Integer.toString(maxEval)+"Eval"+nameProblem+"_"+Integer.toString(
			 * numObj)+"OBJ_Population_"+Integer.toString(populationSize)+"_EXC"+i+".tsv"))
			 * .print();
			 */

			// JMetalLogger.logger.info("Total execution time: " + computingTime + "ms");

			printFinalSolutionSet(population);
			if (!referenceParetoFront.equals("")) {
				// printQualityIndicators(population, referenceParetoFront) ;
				indice = new InvertedGenerationalDistance(referenceParetoFront, 2.0);
				double IGD = indice.evaluate(population);
				igds.add(IGD);
				String now = problemName + ";" + String.valueOf(IGD) + '\n';
				File file = new File(execIdentifier + ".txt");
				FileWriter fr = new FileWriter(file, true);
				fr.write(now);
				fr.close();
				System.out.print(now);
				System.out.println("End of execution " + Integer.toString(i + 1) + " of "
						+ Integer.toString(numExecution));
			}
		}

		// Original problemNAme:
		// String ProblemNAme =
		// "/home/joe/MESTRADO_LINUX/EXPERIMENTOS_NSGA2/"+surrogate+algoritmo+"_"+nameProblem+"_"+Integer.toString(numObj)+"_Objectivos"+Integer.toString(maxEval)+"Eval_Population_"+Integer.toString(populationSize);
		String ProblemNAme = "../../EXPERIMENTOS_NSGA2/" + surrogate + algoritmo + "_" + problemName
				+ "_" + Integer.toString(numObj) + "_Objectivos" + Integer.toString(maxEval)
				+ "Eval_Population_" + Integer.toString(populationSize);

		user userObject = new user(ProblemNAme, ProblemNAme, new ArrayList<>(), igds);
		ArrayList SwarmInicio = http("http://127.0.0.1:5000/save", userObject);
	}

	public static Problem<DoubleSolution> getProblem(String prob, int nObj) {
		int k = -1;
		Problem<DoubleSolution> problem = null;
		if (prob.startsWith("DTLZ")) {
			if(nObj == 3)
				k = 12;
			else if(nObj == 5)
				k = 14;
			else if(nObj == 10)
				k = 19;
			switch (prob) {
			case "DTLZ1":
				problem = new DTLZ1(k, nObj);
				break;
			case "DTLZ2":
				problem = new DTLZ2(k, nObj);
				break;
			case "DTLZ3":
				problem = new DTLZ3(k, nObj);
				break;
			case "DTLZ4":
				problem = new DTLZ4(k, nObj);
				break;
			case "DTLZ5":
				problem = new DTLZ5(k, nObj);
				break;
			case "DTLZ6":
				problem = new DTLZ6(k, nObj);
				break;
			case "DTLZ7":
				problem = new DTLZ7(k, nObj);
				break;
			default:
				problem = null;
				break;
			}
			return problem;
		} else {
			if(nObj == 3)
				k = 4;
			else if(nObj == 5)
				k = 6;
			else if(nObj == 10)
				k = 9;
			switch (prob) {
			case "WFG1":
				problem = new WFG1(k, 10, nObj);
				break;
			case "WFG2":
				problem = new WFG2(k, 10, nObj);
				break;
			case "WFG3":
				problem = new WFG3(k, 10, nObj);
				break;
			case "WFG4":
				problem = new WFG4(k, 10, nObj);
				break;
			case "WFG5":
				problem = new WFG5(k, 10, nObj);
				break;
			case "WFG6":
				problem = new WFG6(k, 10, nObj);
				break;
			case "WFG7":
				problem = new WFG7(k, 10, nObj);
				break;
			case "WFG8":
				problem = new WFG8(k, 10, nObj);
				break;
			case "WFG9":
				problem = new WFG9(k, 10, nObj);
				break;
			default:
				problem = null;
				break;
			}
			return problem;
		}
	}

	public static ArrayList http(String url, user userObject) {
		JSONObject json = new JSONObject();
		ObjectMapper mapper = new ObjectMapper();
		String jsonInString = null;
		try {

			// Convert object to JSON string
			jsonInString = mapper.writeValueAsString(userObject);

			// Convert object to JSON string and pretty print
			jsonInString = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(userObject);

		} catch (JsonGenerationException e) {
			e.printStackTrace();
		} catch (JsonMappingException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		retorno userA = new retorno();
		try (CloseableHttpClient httpClient = HttpClientBuilder.create().build()) {
			HttpPost request = new HttpPost(url);
			StringEntity params = new StringEntity(jsonInString);
			request.addHeader("content-type", "application/json");
			request.setEntity(params);
			HttpResponse result = httpClient.execute(request);

			String json1 = EntityUtils.toString(result.getEntity(), "UTF-8");

			userA = mapper.readValue(json1, retorno.class);

		} catch (IOException ex) {
			System.out.println(ex.getMessage());
		}
		return userA.getRetorno();
	}

}
