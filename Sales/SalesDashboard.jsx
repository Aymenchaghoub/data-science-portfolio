import React, { useState, useMemo, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Filter, TrendingUp, DollarSign, Package, MapPin, Upload, AlertCircle } from 'lucide-react';

const SalesDashboard = () => {
  const [salesData, setSalesData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState('Toutes');
  const [selectedProduct, setSelectedProduct] = useState('Tous');

  // Charger le fichier CSV
  useEffect(() => {
    const loadCSV = async () => {
      try {
        setLoading(true);
        const response = await window.fs.readFile('sales_data_sample.csv', { encoding: 'utf8' });
        
        // Parser le CSV manuellement
        const lines = response.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        
        const data = lines.slice(1).map(line => {
          const values = line.split(',');
          const row = {};
          
          headers.forEach((header, index) => {
            const value = values[index]?.trim();
            
            // Détection et conversion des colonnes
            if (header.includes('date')) {
              row.date = value;
            } else if (header.includes('region') || header.includes('région')) {
              row.region = value;
            } else if (header.includes('product') || header.includes('produit') || header.includes('item')) {
              row.product = value;
            } else if (header.includes('sales') || header.includes('ventes') || header.includes('revenue') || header.includes('ca')) {
              row.sales = parseFloat(value) || 0;
            } else if (header.includes('profit') || header.includes('bénéfice')) {
              row.profit = parseFloat(value) || 0;
            } else if (header.includes('units') || header.includes('unités') || header.includes('quantity') || header.includes('quantité')) {
              row.units = parseFloat(value) || 0;
            }
          });
          
          return row;
        }).filter(row => row.date && row.region && row.product);
        
        setSalesData(data);
        setError(null);
      } catch (err) {
        console.error('Erreur lors du chargement du CSV:', err);
        setError('Impossible de charger le fichier sales_data_sample.csv. Assurez-vous qu\'il est bien chargé dans la conversation.');
      } finally {
        setLoading(false);
      }
    };

    loadCSV();
  }, []);

  const regions = useMemo(() => 
    ['Toutes', ...new Set(salesData.map(d => d.region))].filter(Boolean),
    [salesData]
  );
  
  const products = useMemo(() => 
    ['Tous', ...new Set(salesData.map(d => d.product))].filter(Boolean),
    [salesData]
  );

  // Filtrer les données
  const filteredData = useMemo(() => {
    return salesData.filter(item => {
      const regionMatch = selectedRegion === 'Toutes' || item.region === selectedRegion;
      const productMatch = selectedProduct === 'Tous' || item.product === selectedProduct;
      return regionMatch && productMatch;
    });
  }, [selectedRegion, selectedProduct, salesData]);

  // Calcul des KPIs
  const totalSales = filteredData.reduce((sum, item) => sum + (item.sales || 0), 0);
  const totalProfit = filteredData.reduce((sum, item) => sum + (item.profit || 0), 0);
  const totalUnits = filteredData.reduce((sum, item) => sum + (item.units || 0), 0);
  const profitMargin = totalSales > 0 ? ((totalProfit / totalSales) * 100).toFixed(1) : 0;

  // Ventes par mois
  const salesByMonth = useMemo(() => {
    const monthMap = {};
    filteredData.forEach(item => {
      try {
        const date = new Date(item.date);
        const month = date.toLocaleDateString('fr-FR', { month: 'short', year: '2-digit' });
        if (!monthMap[month]) {
          monthMap[month] = { month, sales: 0, profit: 0, sortKey: date.getTime() };
        }
        monthMap[month].sales += item.sales || 0;
        monthMap[month].profit += item.profit || 0;
      } catch (e) {
        console.error('Erreur de parsing de date:', item.date);
      }
    });
    return Object.values(monthMap).sort((a, b) => a.sortKey - b.sortKey);
  }, [filteredData]);

  // Top produits
  const topProducts = useMemo(() => {
    const productMap = {};
    filteredData.forEach(item => {
      if (!productMap[item.product]) {
        productMap[item.product] = { product: item.product, sales: 0, units: 0 };
      }
      productMap[item.product].sales += item.sales || 0;
      productMap[item.product].units += item.units || 0;
    });
    return Object.values(productMap)
      .sort((a, b) => b.sales - a.sales)
      .slice(0, 5);
  }, [filteredData]);

  // Ventes par région
  const salesByRegion = useMemo(() => {
    const regionMap = {};
    filteredData.forEach(item => {
      if (!regionMap[item.region]) {
        regionMap[item.region] = { region: item.region, value: 0 };
      }
      regionMap[item.region].value += item.sales || 0;
    });
    return Object.values(regionMap);
  }, [filteredData]);

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  // États de chargement et d'erreur
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Chargement des données...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-6">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-800 mb-2 text-center">Erreur de chargement</h2>
          <p className="text-gray-600 text-center mb-4">{error}</p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-gray-700 mb-2"><strong>Instructions :</strong></p>
            <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
              <li>Assurez-vous que le fichier s'appelle exactement <code className="bg-gray-100 px-1 rounded">sales_data_sample.csv</code></li>
              <li>Téléchargez-le dans cette conversation</li>
              <li>Rechargez la page</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  if (salesData.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-6">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
          <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-800 mb-2">Aucune donnée trouvée</h2>
          <p className="text-gray-600">Le fichier CSV est vide ou mal formaté.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* En-tête */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">📊 Tableau de Bord des Ventes</h1>
          <p className="text-gray-600">Analyse des performances commerciales • {salesData.length} transactions chargées</p>
        </div>

        {/* Filtres */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Filter className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-800">Filtres</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Région</label>
              <select
                value={selectedRegion}
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {regions.map(region => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Produit</label>
              <select
                value={selectedProduct}
                onChange={(e) => setSelectedProduct(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {products.map(product => (
                  <option key={product} value={product}>{product}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-xl shadow-md p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Chiffre d'affaires</p>
                <p className="text-2xl font-bold text-gray-800">{totalSales.toLocaleString('fr-FR')} €</p>
              </div>
              <DollarSign className="w-10 h-10 text-blue-500 opacity-80" />
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-6 border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Profit total</p>
                <p className="text-2xl font-bold text-gray-800">{totalProfit.toLocaleString('fr-FR')} €</p>
              </div>
              <TrendingUp className="w-10 h-10 text-green-500 opacity-80" />
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Unités vendues</p>
                <p className="text-2xl font-bold text-gray-800">{totalUnits.toLocaleString('fr-FR')}</p>
              </div>
              <Package className="w-10 h-10 text-orange-500 opacity-80" />
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-6 border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Marge bénéficiaire</p>
                <p className="text-2xl font-bold text-gray-800">{profitMargin}%</p>
              </div>
              <MapPin className="w-10 h-10 text-purple-500 opacity-80" />
            </div>
          </div>
        </div>

        {/* Graphiques */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Évolution mensuelle */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Évolution des ventes et profits</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={salesByMonth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value) => `${value.toLocaleString('fr-FR')} €`} />
                <Legend />
                <Line type="monotone" dataKey="sales" stroke="#3b82f6" name="Ventes" strokeWidth={2} />
                <Line type="monotone" dataKey="profit" stroke="#10b981" name="Profit" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Répartition par région */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Répartition par région</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={salesByRegion}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ region, percent }) => `${region} (${(percent * 100).toFixed(0)}%)`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {salesByRegion.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value.toLocaleString('fr-FR')} €`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top produits */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Top 5 des produits</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topProducts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="product" />
              <YAxis />
              <Tooltip formatter={(value) => `${value.toLocaleString('fr-FR')}`} />
              <Legend />
              <Bar dataKey="sales" fill="#3b82f6" name="Ventes (€)" />
              <Bar dataKey="units" fill="#10b981" name="Unités vendues" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default SalesDashboard;