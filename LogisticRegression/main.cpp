#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>
#include "Eigen\Core"
#include <array>
#include <time.h>
#include <assert.h>
//#include <cuda.h>

using namespace Eigen;
using namespace std;

#define INPUT_DIMENSION 4
#define SAMPLES_SIZE 1000
#define NUMBER_OF_TESTS 10000

//so we can easily control precission
typedef double MReal;

typedef Matrix<MReal, INPUT_DIMENSION, 1> PVector;
typedef Matrix<MReal, 1, INPUT_DIMENSION> PVectorT;
typedef Matrix<MReal, Dynamic,1> ObservationsVector;

typedef Matrix<MReal, INPUT_DIMENSION, INPUT_DIMENSION> HessianMatrix;
typedef Matrix<MReal, Dynamic, INPUT_DIMENSION > PVMatrix;
typedef Matrix<MReal, INPUT_DIMENSION, Dynamic > PVMatrixT;



class FunctionLogLikelyhood
{
	PVMatrix allx;
	PVMatrixT allxT;
	ObservationsVector	 y;
public:
	PVector TestBetas;

	// don't forget, first value of x (or last) needs to be 1
	void GenerateTestData()
	{
		allx = PVMatrix(SAMPLES_SIZE, INPUT_DIMENSION);
		y	 = ObservationsVector(SAMPLES_SIZE,1);
		
		TestBetas = PVector::Random();

		for (unsigned int i = 0; i < SAMPLES_SIZE; i++)
		{
			PVector genvec = PVector::Random();
		
			genvec(0, 0) = 1.f;
			y(i) = probabilityFunction(genvec, TestBetas);
			allx.row(i) = genvec;
		}
		allxT = allx.transpose();
	}
	// populate allx and y
	void Load()
	{

	}
	//p = e^(x * beta)/(1 + e^(x * beta)) 
	MReal inline probabilityFunction(const PVector& x, const PVector& betas){return 1. / (1.f + exp(-x.dot(betas)));}

	//get the log likelyhood at Beta
	MReal inline Eval(const PVector& betas){return (1.0 / (1.0 + exp(-(allx * betas).array())) - y.array()).matrix().squaredNorm();}
	
	//get log likelyhood gradient at beta
	void Gradient(const PVector& beta, PVector& gradientOut)
	{
		const ObservationsVector Values = (1.0 / (exp(-(allx *beta).array()) + 1.0)).matrix() - y;
		gradientOut = allxT*Values;
	}

};

FunctionLogLikelyhood g_function;

MReal line_searchAlpha(const PVector& dir, const PVector& betas)
{
	//new xk test
	MReal c1 = 0.0001;
	MReal alpha = 1.0f;
	MReal tau = 0.5f;
	MReal valueAtNXK = 0.f;
	MReal Wolf1Condition = 0.f;
	do
	{
		
		PVector gradientOut;
		valueAtNXK = g_function.Eval(betas + alpha * dir);
		g_function.Gradient(betas, gradientOut);
		PVectorT gradientTrans = gradientOut.transpose();
		MReal temp = gradientTrans * dir;
		Wolf1Condition = g_function.Eval(betas) + c1*alpha*(temp);
		
		if (valueAtNXK < Wolf1Condition)
			break;

		alpha = alpha * tau;

	} while (valueAtNXK > Wolf1Condition);

	return alpha;
}


void main()
{
	omp_set_num_threads(16);
	Eigen::initParallel();
	srand(time(NULL));
	FILE * f = fopen("tests.txt", "w+");
	unsigned int test = 0;
	unsigned int fails = 0;
	for (; test < NUMBER_OF_TESTS; test++)
	{
		g_function.GenerateTestData();

		PVector  results;

		HessianMatrix BKinv = HessianMatrix::Identity();

		// we start with a random initial estimation
		PVector Betas = PVector::Random();

		const unsigned int MAX_NUM_ITERATIONS = 100;

		MReal gradientNorm = 1.f;
		MReal firstStepAlpha = 0.05f;
		unsigned int iter = 0;
		for (; iter < MAX_NUM_ITERATIONS && gradientNorm >= 0.001; iter++)
		{
			PVector GradientK;
			g_function.Gradient(Betas, GradientK);
			assert(GradientK.allFinite());
			gradientNorm = GradientK.norm();
			//Compute the search direction
			PVector PK = -1 * BKinv * GradientK;

			MReal phi = GradientK.dot(PK);
			
			if (phi > 0) // hessian not positive definite
			{
				BKinv = HessianMatrix::Identity();
				PK = -1 * GradientK;
			}

			MReal alpha = (iter==0)? firstStepAlpha : line_searchAlpha(PK, Betas);
			//Update estimation of beta
			Betas = Betas + alpha * PK;
			assert(Betas.allFinite());
			// get the sk -> this is Betas_(k+1) - Betas_k 
			PVector SK = alpha * PK;

			PVector GradientK1;
			g_function.Gradient(Betas, GradientK1);
			assert(GradientK1.allFinite());
			// get YK  -> this is Gradient_(k+1) - Gradient_k
			PVector YK = GradientK1 - GradientK;


			PVectorT YKT = YK.transpose();
			PVectorT SKT = SK.transpose();
			//Update BK
			MReal YKT_SK = YKT.dot(SK);

			if (YKT_SK == 0.0f)
				continue;

		    //Update BKinv
			MReal SKT_YK = SKT*YK;
			MReal SKT_YKSQ = SKT_YK* SKT_YK;

			BKinv = BKinv + ((SKT_YK + (YKT*BKinv)*YK)*(SK*SKT)) / (SKT_YK*SKT_YK) - (((BKinv*YK)*SKT) + ((SK*YKT)*BKinv)) / SKT_YK;
		}
		if (iter == MAX_NUM_ITERATIONS)
			fails++;

		printf("----------------# %d #---------------\n", test);
		fprintf(f, "----------------# %d #---------------\n",test);
		fprintf(f, "iter:%d error: %f\n", iter, (g_function.TestBetas - Betas).norm());
	}
	fprintf(f, "NUMBER OF FAILS: %d", fails);
	fclose(f);
}