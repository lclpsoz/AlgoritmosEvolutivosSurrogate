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

		int numExecution = 20;
		String metodos[] = { "Batch", "Online" };
		String tagProblems[] = { "DTLZ", "WFG" };
		int numObj = 3;

		// Limits number of version of problems to be simulated.
		int startVersionProblems = 1;
		int maxVersionsProblems = 10;

//		case 0: "NO-SURROGATE";
//		case 1: "SVM";
//		case 2: "RAMDOMFOREST";
//		case 3: "TREE";
//		case 4: "LSTM";
//		case 5: "RNN";
//		case 6: "RANDOM";
		
		int indsClassSurrogates[] = { 4 };
		
		int nodesHiddenLayerRNN = 92;
		int numOfEpochs = 500;
		

		for (int indClassSurrogate : indsClassSurrogates) {
			for (String tagProblem : tagProblems) {

				int amntOfVersions = Math.min(7, maxVersionsProblems);
				if (tagProblem == "WFG")
					amntOfVersions = Math.min(9, maxVersionsProblems);
				for (int p = startVersionProblems; p <= amntOfVersions; p++) {
					for (String metodo : metodos) {

						Problem<DoubleSolution> problem;
						Algorithm<List<DoubleSolution>> algorithm;
						CrossoverOperator<DoubleSolution> crossover;
						MutationOperator<DoubleSolution> mutation;
						SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;
						String referenceParetoFront = "";
						InvertedGenerationalDistance indice;
						String classifierSurrogate = classificador(indClassSurrogate);
						// If Neural Network, the amount of nodes and epochs are
						// integrated in the surrogate name.
						if(indClassSurrogate == 4 || indClassSurrogate == 5)
							classifierSurrogate += 	"_" + Integer.toString(nodesHiddenLayerRNN) +
													"_" + Integer.toString(numOfEpochs);
						ArrayList igds = new ArrayList<>();
						String algoritmo = null;

						String nameProblem = tagProblem + Integer.toString(p);

						int maxEval = 10000;
						int populationSize = -1;
						String surrogate = "Surrogate_" + classifierSurrogate + "_" + metodo;
						boolean online = false;
						boolean noSurrogate = false;
						if(indClassSurrogate == 0)
							noSurrogate = false;

						if (metodo.equals("Online"))
							online = true;

						// Identifier of the current simulation!
						String execIdentifier = "out_NSGAII45_IGD_" + tagProblem + "_" + surrogate + "_" + "Obj-"
								+ Integer.toString(numObj) + "_" + "EvalPopulation-" + Integer.toString(maxEval) + "_"
								+ "PopulationSize-" + Integer.toString(populationSize) + "_" + "timeStamp-"
								+ String.valueOf(tNow);
						System.out.println("----------| " + execIdentifier + " |----------");

						for (int i = 0; i < numExecution; i++) {
							String problemName;
							if (args.length == 1) {
								problemName = args[0];
							} else if (args.length == 2) {
								problemName = args[0];
								referenceParetoFront = args[1];
							} else {
								if (nameProblem.startsWith("DTLZ"))
									problemName = "org.uma.jmetal.problem.multiobjective.dtlz." + nameProblem;
								else
									problemName = "org.uma.jmetal.problem.multiobjective.wfg." + nameProblem;
								referenceParetoFront = "";
							}

							// problem = ProblemUtils.<DoubleSolution> loadProblem(problemName);
							// problem = problem.createSolution();
							problem = getProblem(nameProblem, numObj);

							referenceParetoFront = "../jmetal-problem/src/test/resources/pareto_fronts/" + nameProblem
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
									classifierSurrogate + "_" + nameProblem + "_" + metodo,
									tagProblem,
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
									online, noSurrogate);

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
								String now = tagProblem + String.valueOf(p) + ";" + String.valueOf(IGD) + '\n';
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
						String ProblemNAme = "../../EXPERIMENTOS_NSGA2/" + surrogate + algoritmo + "_" + nameProblem
								+ "_" + Integer.toString(numObj) + "_Objectivos" + Integer.toString(maxEval)
								+ "Eval_Population_" + Integer.toString(populationSize);

						user userObject = new user(ProblemNAme, ProblemNAme, new ArrayList<>(), igds);
						ArrayList SwarmInicio = http("http://127.0.0.1:5000/save", userObject);
					}
				}
			}
		}
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

	public static String classificador(int index) {
		String classificador = null;
		switch (index) {
		case 1:
			classificador = "SVM";
			break;
		case 2:
			classificador = "RAMDOMFOREST";
			break;
		case 3:
			classificador = "TREE";
			break;
		case 4:
			classificador = "LSTM";
			break;
		case 5:
			classificador = "RNN";
			break;
		case 6:
			classificador = "RANDOM";
			break;
		default:
			classificador = "NO-SURROGATE";
			break;
		}
		return classificador;
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
